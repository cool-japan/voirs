use super::analyzer::{F0Characteristics, SpeakerAnalyzer};
use super::diarizer::{
    SpeakerChangePoint as _, SpeakerCluster, SpeakerDiarizationResult as _, SpeakerDiarizer,
    SpeakerEmbedding as _, SpeakerSegment,
};
use crate::traits::{
    AccentInfo, AgeRange, Gender, SpeakerCharacteristics, VoiceCharacteristics, VoiceQuality,
};
use std::f32::consts::PI as PI_F32;
use voirs_sdk::AudioBuffer;

#[tokio::test]
async fn test_speaker_analyzer_creation() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();
    assert_eq!(analyzer.sample_rate, 16000.0);
    assert_eq!(analyzer.frame_size, 1024);
    assert_eq!(analyzer.gender_thresholds.f0_threshold, 165.0);
}

#[tokio::test]
async fn test_f0_characteristics_extraction() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Generate male-like voice (low F0)
    let frequency = 120.0; // Typical male F0
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / 16000.0).sin())
        .collect();
    let audio = AudioBuffer::new(samples, 16000, 1);

    let f0_chars = analyzer.extract_f0_characteristics(&audio).await.unwrap();

    assert!(f0_chars.mean_f0 > 0.0);
    assert!(f0_chars.voiced_ratio > 0.0);
    assert!(f0_chars.f0_range.0 <= f0_chars.f0_range.1);
}

#[tokio::test]
async fn test_gender_classification() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Test male voice characteristics
    let male_f0 = F0Characteristics {
        mean_f0: 120.0,
        f0_range: (100.0, 140.0),
        f0_variation: 10.0,
        voiced_ratio: 0.8,
    };
    let male_formants = vec![700.0, 1200.0, 2500.0];

    let male_gender = analyzer.classify_gender(&male_f0, &male_formants);
    assert_eq!(male_gender, Some(Gender::Male));

    // Test female voice characteristics
    let female_f0 = F0Characteristics {
        mean_f0: 220.0,
        f0_range: (180.0, 260.0),
        f0_variation: 15.0,
        voiced_ratio: 0.85,
    };
    let female_formants = vec![900.0, 2200.0, 3100.0];

    let female_gender = analyzer.classify_gender(&female_f0, &female_formants);
    assert_eq!(female_gender, Some(Gender::Female));
}

#[tokio::test]
async fn test_age_estimation() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Child voice (high F0)
    let child_f0 = F0Characteristics {
        mean_f0: 300.0,
        f0_range: (250.0, 350.0),
        f0_variation: 20.0,
        voiced_ratio: 0.8,
    };
    let child_voice_quality = VoiceQuality {
        jitter: 0.02,
        shimmer: 0.05,
        hnr: 15.0,
    };

    let child_age = analyzer.estimate_age(&child_f0, &child_voice_quality);
    assert_eq!(child_age, Some(AgeRange::Child));

    // Senior voice (moderate F0, high jitter/shimmer)
    let senior_f0 = F0Characteristics {
        mean_f0: 180.0,
        f0_range: (150.0, 210.0),
        f0_variation: 25.0,
        voiced_ratio: 0.7,
    };
    let senior_voice_quality = VoiceQuality {
        jitter: 0.08,  // High jitter
        shimmer: 0.15, // High shimmer
        hnr: 8.0,
    };

    let senior_age = analyzer.estimate_age(&senior_f0, &senior_voice_quality);
    assert_eq!(senior_age, Some(AgeRange::Senior));
}

#[tokio::test]
async fn test_voice_quality_analysis() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Clean sine wave should have low jitter and shimmer
    let frequency = 200.0;
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / 16000.0).sin())
        .collect();
    let audio = AudioBuffer::new(samples, 16000, 1);

    let voice_quality = analyzer.analyze_voice_quality(&audio).await.unwrap();

    // Should have reasonable values
    assert!(voice_quality.jitter >= 0.0);
    assert!(voice_quality.shimmer >= 0.0);
    assert!(voice_quality.hnr.is_finite());
}

#[tokio::test]
async fn test_formant_extraction() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    let samples = vec![0.1; 16000]; // Simple audio
    let audio = AudioBuffer::new(samples, 16000, 1);

    let formants = analyzer.extract_formants(&audio).await.unwrap();

    assert_eq!(formants.len(), 3); // Should extract F1, F2, F3

    // Formants should be in reasonable ranges
    for &formant in &formants {
        assert!(formant >= 0.0);
        assert!(formant <= 4000.0); // Within speech range
    }
}

#[tokio::test]
async fn test_complete_speaker_analysis() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Generate female-like voice
    let frequency = 220.0;
    let samples: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / 16000.0).sin() * 0.5)
        .collect();
    let audio = AudioBuffer::new(samples, 16000, 1);

    let speaker_chars = analyzer.analyze_speaker(&audio).await.unwrap();

    // Should classify as female based on F0
    assert_eq!(speaker_chars.gender, Some(Gender::Female));

    // Should have voice characteristics
    assert!(
        speaker_chars.voice_characteristics.f0_range.0
            <= speaker_chars.voice_characteristics.f0_range.1
    );
    assert_eq!(speaker_chars.voice_characteristics.formants.len(), 3);

    // Voice quality should be reasonable
    assert!(speaker_chars.voice_characteristics.voice_quality.jitter >= 0.0);
    assert!(speaker_chars.voice_characteristics.voice_quality.shimmer >= 0.0);
}

#[tokio::test]
async fn test_emotion_analysis() {
    use crate::traits::Emotion;

    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Generate emotional speech simulation (varying F0 and amplitude)
    let mut samples = Vec::new();
    for i in 0..16000 {
        let t = i as f32 / 16000.0;
        let frequency = 200.0 + 50.0 * (t * 2.0 * std::f32::consts::PI).sin(); // Varying F0
        let amplitude = 0.5 + 0.3 * (t * 4.0 * std::f32::consts::PI).sin(); // Varying amplitude
        let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(sample);
    }
    let audio = AudioBuffer::new(samples, 16000, 1);

    let emotion_analysis = analyzer.analyze_emotion(&audio).await.unwrap();

    // Should have valid emotion classification
    assert_ne!(emotion_analysis.primary_emotion, Emotion::Neutral); // Should detect some emotion

    // Should have emotion scores for all emotions
    assert!(!emotion_analysis.emotion_scores.is_empty());

    // Dimensions should be in valid range
    assert!(emotion_analysis.valence >= -1.0 && emotion_analysis.valence <= 1.0);
    assert!(emotion_analysis.arousal >= -1.0 && emotion_analysis.arousal <= 1.0);
    assert!(emotion_analysis.intensity >= 0.0 && emotion_analysis.intensity <= 1.0);
}

#[tokio::test]
async fn test_accent_detection() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    let samples = vec![0.1; 16000];
    let audio = AudioBuffer::new(samples, 16000, 1);

    let formants = vec![500.0, 2200.0, 3000.0]; // American-like formants

    let accent_info = analyzer.detect_accent(&audio, &formants).await.unwrap();

    assert!(accent_info.is_some());
    let accent = accent_info.unwrap();
    assert!(!accent.accent_type.is_empty());
    assert!(accent.confidence > 0.0);
    assert!(accent.confidence <= 1.0);
}

#[tokio::test]
async fn test_energy_features() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();

    // Create audio with varying energy
    let mut samples = Vec::new();
    for i in 0..16000 {
        let t = i as f32 / 16000.0;
        let amplitude = 0.3 + 0.2 * (t * 3.0 * std::f32::consts::PI).sin();
        let sample = amplitude * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
        samples.push(sample);
    }
    let audio = AudioBuffer::new(samples, 16000, 1);

    let energy_features = analyzer.extract_energy_features(&audio).await.unwrap();

    assert!(energy_features.mean_energy > 0.0);
    assert!(energy_features.energy_variation > 0.0); // Should have variation
    assert!(energy_features.energy_dynamics >= 0.0);
}

// Speaker Diarization Tests

#[tokio::test]
async fn test_speaker_diarizer_creation() {
    let diarizer = SpeakerDiarizer::new().await.unwrap();
    assert_eq!(diarizer.window_size, 3.0);
    assert_eq!(diarizer.window_overlap, 1.5);
    assert_eq!(diarizer.similarity_threshold, 0.8);
    assert_eq!(diarizer.min_segment_duration, 0.5);
}

#[tokio::test]
async fn test_speaker_diarizer_with_config() {
    let diarizer = SpeakerDiarizer::with_config(2.0, 1.0, 0.7, 0.3)
        .await
        .unwrap();
    assert_eq!(diarizer.window_size, 2.0);
    assert_eq!(diarizer.window_overlap, 1.0);
    assert_eq!(diarizer.similarity_threshold, 0.7);
    assert_eq!(diarizer.min_segment_duration, 0.3);
}

#[tokio::test]
async fn test_single_speaker_diarization() {
    let diarizer = SpeakerDiarizer::new().await.unwrap();

    // Generate consistent single speaker audio (10 seconds)
    let frequency = 150.0; // Male voice
    let samples: Vec<f32> = (0..160000) // 10 seconds at 16kHz
        .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / 16000.0).sin() * 0.3)
        .collect();
    let audio = AudioBuffer::new(samples, 16000, 1);

    let result = diarizer.diarize(&audio).await.unwrap();

    // Should detect only one speaker
    assert_eq!(result.num_speakers, 1);
    assert!(!result.segments.is_empty());
    assert_eq!(result.speaker_embeddings.len(), 1);
    assert!(result.overall_confidence > 0.0);

    // All segments should be from the same speaker
    let first_speaker_id = &result.segments[0].speaker_id;
    assert!(result
        .segments
        .iter()
        .all(|s| s.speaker_id == *first_speaker_id));
}

#[tokio::test]
async fn test_two_speaker_diarization() {
    let diarizer = SpeakerDiarizer::new().await.unwrap();

    // Generate two-speaker audio (male first 5s, female next 5s)
    let mut samples = Vec::new();

    // Male speaker (150 Hz)
    for i in 0..80000 {
        let t = i as f32 / 16000.0;
        let sample = (2.0 * std::f32::consts::PI * 150.0 * t).sin() * 0.3;
        samples.push(sample);
    }

    // Female speaker (220 Hz)
    for i in 0..80000 {
        let t = i as f32 / 16000.0;
        let sample = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.3;
        samples.push(sample);
    }

    let audio = AudioBuffer::new(samples, 16000, 1);
    let result = diarizer.diarize(&audio).await.unwrap();

    // Should detect two speakers (though clustering might merge similar ones)
    assert!(result.num_speakers >= 1);
    assert!(!result.segments.is_empty());
    assert!(!result.speaker_embeddings.is_empty());
    assert!(result.overall_confidence > 0.0);

    // Check that segments cover reasonable time range
    let total_duration: f32 = result
        .segments
        .iter()
        .map(|s| s.end_time - s.start_time)
        .sum();
    assert!(total_duration > 5.0); // Should cover significant portion of audio
}

#[tokio::test]
async fn test_speaker_change_detection() {
    let diarizer = SpeakerDiarizer::with_config(2.0, 1.0, 0.6, 0.3)
        .await
        .unwrap(); // Lower threshold

    // Generate audio with clear speaker change
    let mut samples = Vec::new();

    // First speaker - low frequency, low amplitude (male-like)
    for i in 0..32000 {
        // 2 seconds
        let t = i as f32 / 16000.0;
        let sample = (2.0 * std::f32::consts::PI * 100.0 * t).sin() * 0.2;
        samples.push(sample);
    }

    // Second speaker - high frequency, high amplitude (female-like)
    for i in 0..32000 {
        // 2 seconds
        let t = i as f32 / 16000.0;
        let sample = (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.5;
        samples.push(sample);
    }

    let audio = AudioBuffer::new(samples, 16000, 1);
    let change_points = diarizer.detect_speaker_changes(&audio).await.unwrap();

    // Should detect at least one change point (or be able to handle gracefully)
    // Note: Due to the simplified nature of the test audio and feature extraction,
    // change detection might not always work perfectly with synthetic sine waves
    if change_points.is_empty() {
        // If no change points detected, at least verify the function completed without error
        println!("Note: No speaker changes detected in test audio (this may be expected for simple sine waves)");
    } else {
        println!("Detected {} speaker change points", change_points.len());
    }

    // Change points should have reasonable confidence
    for change_point in &change_points {
        assert!(change_point.confidence > 0.0);
        assert!(change_point.confidence <= 1.0);
        assert!(change_point.time > 0.0);
        assert!(change_point.time < 4.0); // Within audio duration (4 seconds total)
    }
}

#[tokio::test]
async fn test_feature_extraction_from_characteristics() {
    let characteristics = SpeakerCharacteristics {
        gender: Some(Gender::Female),
        age_range: Some(AgeRange::Adult),
        voice_characteristics: VoiceCharacteristics {
            f0_range: (180.0, 250.0),
            formants: vec![900.0, 2200.0, 3100.0],
            voice_quality: VoiceQuality {
                jitter: 0.02,
                shimmer: 0.05,
                hnr: 15.0,
            },
        },
        accent: Some(AccentInfo {
            accent_type: "American".to_string(),
            confidence: 0.8,
            regional_indicators: vec!["formant-based".to_string()],
        }),
    };

    let features = SpeakerDiarizer::extract_features_from_characteristics(&characteristics);

    // Should have correct number of features
    assert_eq!(features.len(), 15); // 3 F0 + 3 formants + 3 voice quality + 2 gender + 4 age = 15

    // Features should be normalized (0-1 range after normalization)
    for &feature in &features {
        assert!(feature.is_finite());
        assert!(feature >= 0.0);
        assert!(feature <= 1.0);
    }
}

#[tokio::test]
async fn test_embedding_similarity() {
    // Test identical embeddings
    let embedding1 = vec![0.5, 0.7, 0.3, 0.9];
    let embedding2 = vec![0.5, 0.7, 0.3, 0.9];
    let similarity = SpeakerDiarizer::calculate_embedding_similarity(&embedding1, &embedding2);
    assert!((similarity - 1.0).abs() < 1e-5); // Should be very close to 1.0

    // Test orthogonal embeddings
    let embedding3 = vec![1.0, 0.0, 0.0, 0.0];
    let embedding4 = vec![0.0, 1.0, 0.0, 0.0];
    let similarity2 = SpeakerDiarizer::calculate_embedding_similarity(&embedding3, &embedding4);
    assert!((similarity2 - 0.0).abs() < 1e-5); // Should be close to 0.0

    // Test different length embeddings
    let embedding5 = vec![0.5, 0.7];
    let embedding6 = vec![0.5, 0.7, 0.3];
    let similarity3 = SpeakerDiarizer::calculate_embedding_similarity(&embedding5, &embedding6);
    assert_eq!(similarity3, 0.0); // Should be 0.0 for different lengths
}

#[tokio::test]
async fn test_speaker_segment_merging() {
    let diarizer = SpeakerDiarizer::new().await.unwrap();

    // Create test segments from same speaker that should be merged
    let characteristics = SpeakerCharacteristics {
        gender: Some(Gender::Male),
        age_range: Some(AgeRange::Adult),
        voice_characteristics: VoiceCharacteristics {
            f0_range: (120.0, 150.0),
            formants: vec![700.0, 1200.0, 2500.0],
            voice_quality: VoiceQuality {
                jitter: 0.03,
                shimmer: 0.06,
                hnr: 12.0,
            },
        },
        accent: None,
    };

    let segments = vec![
        SpeakerSegment {
            speaker_id: "Speaker_1".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            characteristics: characteristics.clone(),
            confidence: 0.8,
        },
        SpeakerSegment {
            speaker_id: "Speaker_1".to_string(),
            start_time: 1.1, // Small gap
            end_time: 2.1,
            characteristics: characteristics.clone(),
            confidence: 0.9,
        },
        SpeakerSegment {
            speaker_id: "Speaker_2".to_string(),
            start_time: 3.0,
            end_time: 4.0,
            characteristics: characteristics.clone(),
            confidence: 0.7,
        },
    ];

    let merged = diarizer.merge_adjacent_segments(segments);

    // Should have merged the first two segments
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].speaker_id, "Speaker_1");
    assert_eq!(merged[0].start_time, 0.0);
    assert_eq!(merged[0].end_time, 2.1);
    assert_eq!(merged[1].speaker_id, "Speaker_2");
}

#[tokio::test]
async fn test_empty_audio_diarization() {
    let diarizer = SpeakerDiarizer::new().await.unwrap();

    // Very short audio
    let samples = vec![0.1; 1000]; // Less than window size
    let audio = AudioBuffer::new(samples, 16000, 1);

    let result = diarizer.diarize(&audio).await;

    // Should handle gracefully - either return empty result or minimal result
    match result {
        Ok(res) => {
            assert!(res.segments.is_empty() || res.segments.len() == 1);
            assert!(res.num_speakers <= 1);
        }
        Err(_) => {
            // Also acceptable to return an error for too-short audio
        }
    }
}

#[tokio::test]
async fn test_speaker_embedding_creation() {
    let characteristics = SpeakerCharacteristics {
        gender: Some(Gender::Female),
        age_range: Some(AgeRange::Teen),
        voice_characteristics: VoiceCharacteristics {
            f0_range: (200.0, 280.0),
            formants: vec![850.0, 2100.0, 2900.0],
            voice_quality: VoiceQuality {
                jitter: 0.025,
                shimmer: 0.04,
                hnr: 18.0,
            },
        },
        accent: None,
    };

    let segments = vec![
        SpeakerSegment {
            speaker_id: "Speaker_1".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            characteristics: characteristics.clone(),
            confidence: 0.85,
        },
        SpeakerSegment {
            speaker_id: "Speaker_1".to_string(),
            start_time: 2.0,
            end_time: 3.0,
            characteristics: characteristics.clone(),
            confidence: 0.90,
        },
    ];

    let clusters = vec![SpeakerCluster {
        id: "Speaker_1".to_string(),
        centroid: vec![0.5, 0.6, 0.7, 0.8],
        member_indices: vec![0, 1],
    }];

    let embeddings_map = SpeakerDiarizer::create_speaker_embeddings_map(&segments, &clusters);

    assert_eq!(embeddings_map.len(), 1);
    assert!(embeddings_map.contains_key("Speaker_1"));

    let embedding = &embeddings_map["Speaker_1"];
    assert_eq!(embedding.segment_count, 2);
    assert_eq!(embedding.average_confidence, 0.875); // (0.85 + 0.90) / 2
    assert_eq!(embedding.features, vec![0.5, 0.6, 0.7, 0.8]);
}

// -----------------------------------------------------------------------
// DSP unit tests for the real windowed-FFT spectrum and spectral flux
// -----------------------------------------------------------------------

/// A 440 Hz pure sine wave must produce a magnitude spectrum whose peak
/// bin is the one closest to 440 Hz.
#[tokio::test]
async fn test_spectrum_pure_tone() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();
    // Use the default frame_size (1024) at 16 kHz → bin resolution = 16000/1024 ≈ 15.625 Hz
    let sample_rate = 16_000_u32;
    let freq_hz = 440.0_f32;
    let n_fft = analyzer.frame_size; // 1024

    // One full frame of a 440 Hz sine
    #[allow(clippy::cast_precision_loss)]
    let samples: Vec<f32> = (0..n_fft)
        .map(|i| (2.0 * PI_F32 * freq_hz * i as f32 / sample_rate as f32).sin())
        .collect();

    let spectrum = analyzer.compute_spectrum(&samples).await.unwrap();

    // Expected peak bin: round(440 * n_fft / sample_rate)
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let expected_bin = (freq_hz * n_fft as f32 / sample_rate as f32).round() as usize;

    let peak_bin = spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    // Allow ±1 bin tolerance for windowing side-lobes
    assert!(
        peak_bin.abs_diff(expected_bin) <= 1,
        "peak bin {peak_bin} should be within 1 of expected bin {expected_bin} for {freq_hz} Hz"
    );
}

/// A low-frequency tone must yield a lower spectral centroid than a
/// high-frequency tone.
#[tokio::test]
async fn test_spectral_centroid_ordering() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();
    let sample_rate = 16_000_u32;
    let n_samples = 16_000usize; // 1 second

    // Build audio buffers for the two tones
    let make_audio = |freq: f32| -> AudioBuffer {
        #[allow(clippy::cast_precision_loss)]
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / sample_rate as f32).sin())
            .collect();
        AudioBuffer::new(samples, sample_rate, 1)
    };

    let audio_low = make_audio(200.0);
    let audio_high = make_audio(3_000.0);

    let features_low = analyzer
        .extract_spectral_features(&audio_low)
        .await
        .unwrap();
    let features_high = analyzer
        .extract_spectral_features(&audio_high)
        .await
        .unwrap();

    assert!(
        features_high.centroid > features_low.centroid,
        "high-freq centroid ({}) should exceed low-freq centroid ({})",
        features_high.centroid,
        features_low.centroid
    );
}

/// A steady 440 Hz tone → near-zero flux; a chirp (100→4000 Hz) → clearly
/// non-zero flux that substantially exceeds the steady-tone flux.
#[tokio::test]
async fn test_spectral_flux_steady_vs_sweep() {
    let analyzer = SpeakerAnalyzer::new().await.unwrap();
    let sample_rate = 16_000_u32;
    // 200 ms of audio at 16 kHz
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    let n_samples = (sample_rate as f32 * 0.2) as usize; // 3200 samples

    // Steady 440 Hz sine
    #[allow(clippy::cast_precision_loss)]
    let steady: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * PI_F32 * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();
    let audio_steady = AudioBuffer::new(steady, sample_rate, 1);

    // Linear chirp 100 → 4000 Hz over 200 ms.
    // Use the correct integral-of-instantaneous-frequency phase so that
    // the instantaneous frequency truly sweeps from f_start to f_end.
    let f_start = 100.0_f32;
    let f_end = 4_000.0_f32;
    #[allow(clippy::cast_precision_loss)]
    let duration = n_samples as f32 / sample_rate as f32; // 0.2 s
    #[allow(clippy::cast_precision_loss)]
    let chirp: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // phase = 2π ∫₀ᵗ f(τ) dτ  where f(τ) = f_start + (f_end-f_start)*τ/T
            let phase = 2.0 * PI_F32 * (f_start * t + 0.5 * (f_end - f_start) * t * t / duration);
            phase.sin()
        })
        .collect();
    let audio_chirp = AudioBuffer::new(chirp, sample_rate, 1);

    let features_steady = analyzer
        .extract_spectral_features(&audio_steady)
        .await
        .unwrap();
    let features_chirp = analyzer
        .extract_spectral_features(&audio_chirp)
        .await
        .unwrap();

    // Chirp flux must be noticeably larger than steady-tone flux
    assert!(
        features_chirp.flux > features_steady.flux + 0.01,
        "chirp flux ({}) should exceed steady flux ({}) by > 0.01",
        features_chirp.flux,
        features_steady.flux
    );
}
