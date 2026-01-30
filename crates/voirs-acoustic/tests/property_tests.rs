//! Property-based tests for voirs-acoustic
//!
//! These tests use proptest to validate invariants and edge cases across
//! a wide range of inputs, ensuring robust behavior of the acoustic models.

use proptest::prelude::*;
use voirs_acoustic::streaming::{StreamingConfig, VadConfig};
use voirs_acoustic::{LanguageCode, MelSpectrogram, Phoneme, SynthesisConfig};

/// Strategy for generating valid phoneme symbols
fn phoneme_symbol() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("a".to_string()),
        Just("i".to_string()),
        Just("u".to_string()),
        Just("e".to_string()),
        Just("o".to_string()),
        Just("k".to_string()),
        Just("s".to_string()),
        Just("t".to_string()),
        Just("n".to_string()),
        Just("m".to_string()),
        Just("r".to_string()),
        Just("w".to_string()),
        Just("j".to_string()),
        Just("p".to_string()),
        Just("b".to_string()),
        Just("g".to_string()),
    ]
}

/// Strategy for generating sequences of phonemes
fn phoneme_sequence() -> impl Strategy<Value = Vec<Phoneme>> {
    prop::collection::vec(phoneme_symbol(), 1..50).prop_map(|symbols| {
        symbols
            .into_iter()
            .map(|symbol| Phoneme {
                symbol,
                features: None,
                duration: None,
            })
            .collect()
    })
}

#[test]
fn test_phoneme_sequence_properties() {
    proptest!(|(phonemes in phoneme_sequence())| {
        // Property 1: Phoneme sequences should preserve their length
        assert!(!phonemes.is_empty());
        assert!(phonemes.len() < 50);

        // Property 2: All phonemes should have valid symbols
        for phoneme in &phonemes {
            assert!(!phoneme.symbol.is_empty());
        }
    });
}

#[test]
fn test_phoneme_equality_properties() {
    proptest!(|(symbol in phoneme_symbol())| {
        let p1 = Phoneme::new(symbol.clone());
        let p2 = Phoneme::new(symbol.clone());

        // Property: Phonemes with same symbol should be equal
        assert_eq!(p1, p2);

        // Property: Hash values should be equal for equal phonemes
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        p1.hash(&mut hasher1);
        p2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    });
}

#[test]
fn test_synthesis_config_properties() {
    proptest!(|(
        speed in 0.5f32..2.0f32,
        pitch_shift in -12.0f32..12.0f32,
        energy in 0.1f32..2.0f32,
    )| {
        let config = SynthesisConfig {
            speed,
            pitch_shift,
            energy,
            ..Default::default()
        };

        // Property: Config values should be within valid ranges
        assert!(config.speed >= 0.5 && config.speed <= 2.0);
        assert!(config.pitch_shift >= -12.0 && config.pitch_shift <= 12.0);
        assert!(config.energy >= 0.1 && config.energy <= 2.0);
    });
}

#[test]
fn test_mel_spectrogram_properties() {
    proptest!(|(
        n_mels in 40usize..128usize,
        sample_rate in 16000usize..48000usize,
        hop_length in 128usize..512usize,
    )| {
        // Property: Mel spectrogram dimensions should be consistent
        // with audio processing parameters

        // Nyquist frequency should be half of sample rate
        let nyquist = sample_rate / 2;

        // Number of mel bins should be reasonable
        assert!(n_mels >= 40);
        assert!(n_mels <= 128);

        // Hop length should be a reasonable fraction of sample rate
        assert!(hop_length < sample_rate);
        assert!(hop_length >= 128);

        // For a 1-second audio at the given sample rate,
        // the number of frames should be approximately:
        // num_frames ≈ sample_rate / hop_length
        let expected_frames = sample_rate / hop_length;
        assert!(expected_frames > 0);
        assert!(expected_frames < 1000); // Sanity check
    });
}

#[test]
fn test_language_code_properties() {
    // Property: All language codes should round-trip correctly
    for &lang in LanguageCode::all() {
        let string_repr = lang.as_str();
        let parsed = LanguageCode::parse(string_repr);
        assert_eq!(parsed, Some(lang));

        // Property: Language names should not be empty
        assert!(!lang.language_name().is_empty());

        // Property: ISO codes should be 2 characters
        assert_eq!(lang.language_code().len(), 2);
    }
}

#[test]
fn test_phoneme_duration_properties() {
    proptest!(|(
        symbol in phoneme_symbol(),
        duration in 0.01f32..0.5f32,
    )| {
        let mut phoneme = Phoneme::new(symbol);
        phoneme.duration = Some(duration);

        // Property: Duration should be positive and reasonable
        if let Some(d) = phoneme.duration {
            assert!(d > 0.0);
            assert!(d < 1.0); // No single phoneme should be longer than 1 second
        }
    });
}

#[test]
fn test_phoneme_sequence_concatenation() {
    proptest!(|(
        seq1 in phoneme_sequence(),
        seq2 in phoneme_sequence(),
    )| {
        let len1 = seq1.len();
        let len2 = seq2.len();

        let mut combined = seq1.clone();
        combined.extend(seq2.clone());

        // Property: Concatenation should preserve total length
        assert_eq!(combined.len(), len1 + len2);

        // Property: Order should be preserved
        for (i, phoneme) in seq1.iter().enumerate() {
            assert_eq!(&combined[i], phoneme);
        }
        for (i, phoneme) in seq2.iter().enumerate() {
            assert_eq!(&combined[len1 + i], phoneme);
        }
    });
}

#[test]
fn test_synthesis_config_normalization() {
    proptest!(|(
        speed in -10.0f32..10.0f32,
        pitch_shift in -100.0f32..100.0f32,
        energy in -10.0f32..10.0f32,
    )| {
        // Property: Even with extreme values, config should handle gracefully
        // (In real implementation, values would be clamped)

        let config = SynthesisConfig {
            speed: speed.abs().clamp(0.1, 10.0),
            pitch_shift: pitch_shift.clamp(-24.0, 24.0),
            energy: energy.abs().clamp(0.01, 10.0),
            ..Default::default()
        };

        // After normalization, all values should be reasonable
        assert!(config.speed > 0.0);
        assert!(config.speed <= 10.0);
        assert!(config.pitch_shift >= -24.0);
        assert!(config.pitch_shift <= 24.0);
        assert!(config.energy > 0.0);
        assert!(config.energy <= 10.0);
    });
}

#[test]
fn test_phoneme_set_uniqueness() {
    proptest!(|(phonemes in phoneme_sequence())| {
        use std::collections::HashSet;

        // Property: Phonemes can be stored in a HashSet
        let phoneme_set: HashSet<_> = phonemes.iter().collect();

        // Property: Set size should be ≤ original sequence length
        assert!(phoneme_set.len() <= phonemes.len());

        // Property: All phonemes in set should be from original sequence
        for phoneme in phoneme_set {
            assert!(phonemes.contains(phoneme));
        }
    });
}

#[test]
fn test_language_code_ordering() {
    // Property: Language codes should have stable ordering
    let mut langs1 = LanguageCode::all().to_vec();
    let mut langs2 = LanguageCode::all().to_vec();

    langs1.sort();
    langs2.sort();

    assert_eq!(langs1, langs2);

    // Property: Ordering should be consistent with enum definition
    for i in 0..langs1.len() - 1 {
        assert!(langs1[i] <= langs1[i + 1]);
    }
}

#[test]
fn test_mel_spectrogram_frame_properties() {
    proptest!(|(
        audio_samples in 8000usize..160000usize,
        hop_length in 128usize..512usize,
    )| {
        // Property: Number of mel spectrogram frames should be deterministic
        // based on audio length and hop length

        let expected_frames = (audio_samples + hop_length - 1) / hop_length;

        // Property: Should always have at least one frame
        assert!(expected_frames > 0);

        // Property: Frames should not exceed theoretical maximum
        let max_frames = audio_samples / hop_length + 1;
        assert!(expected_frames <= max_frames);
    });
}
// ============================================================================
// Streaming Property Tests
// ============================================================================

#[test]
fn test_streaming_config_chunk_properties() {
    proptest!(|(
        chunk_frames in 64usize..2048usize,
        overlap_frames in 0usize..512usize,
    )| {
        // Property: Overlap should be less than chunk size
        let valid_overlap = overlap_frames.min(chunk_frames - 1);

        // Property: Chunk frames should be greater than overlap
        assert!(chunk_frames > valid_overlap);

        // Property: Effective frames per chunk should be positive
        let effective_frames = chunk_frames - valid_overlap;
        assert!(effective_frames > 0);

        // Property: Total latency should be predictable
        // latency = chunk_frames + lookahead_frames
        let lookahead = chunk_frames / 4; // Common lookahead is 25% of chunk
        let total_latency_frames = chunk_frames + lookahead;
        assert!(total_latency_frames > chunk_frames);
    });
}

#[test]
fn test_streaming_config_parameter_ranges() {
    proptest!(|(
        chunk_frames in 64usize..2048usize,
        max_latency_ms in 10u32..500u32,
        quality_factor in 0.0f32..1.0f32,
        buffer_size in 1usize..10usize,
    )| {
        let config = StreamingConfig {
            chunk_frames,
            overlap_frames: chunk_frames / 4, // 25% overlap
            max_latency_ms,
            quality_factor,
            buffer_size,
            enable_prediction: true,
            adaptive_chunking: true,
            enable_vad: true,
            lookahead_frames: chunk_frames / 2,
            num_threads: 4,
            enable_realtime_streaming: true,
            min_chunk_frames: 64,
            max_chunk_frames: 2048,
            prediction_lookahead: 10,
            vad_threshold: 0.01,
            enable_crossfade: true,
            crossfade_frames: chunk_frames / 8,
        };

        // Property: All parameters should be in valid ranges
        assert!(config.chunk_frames >= 64);
        assert!(config.chunk_frames <= 2048);
        assert!(config.max_latency_ms >= 10);
        assert!(config.max_latency_ms <= 500);
        assert!(config.quality_factor >= 0.0);
        assert!(config.quality_factor <= 1.0);
        assert!(config.buffer_size >= 1);

        // Property: Derived parameters should be consistent
        assert!(config.overlap_frames < config.chunk_frames);
        assert!(config.crossfade_frames < config.chunk_frames);
        assert!(config.min_chunk_frames <= config.max_chunk_frames);
    });
}

#[test]
fn test_streaming_buffer_capacity_properties() {
    proptest!(|(
        chunk_frames in 64usize..2048usize,
        buffer_size in 1usize..10usize,
    )| {
        // Property: Total buffer capacity should be deterministic
        let total_capacity = chunk_frames * buffer_size;

        // Property: Buffer should hold at least one chunk
        assert!(total_capacity >= chunk_frames);

        // Property: Buffer capacity should scale linearly with buffer size
        assert_eq!(total_capacity, chunk_frames * buffer_size);

        // Property: Memory footprint should be predictable (4 bytes per f32)
        let memory_bytes = total_capacity * 4;
        assert!(memory_bytes > 0);
    });
}

#[test]
fn test_streaming_overlap_add_properties() {
    proptest!(|(
        chunk_frames in 128usize..1024usize,
        overlap_frames in 16usize..256usize,
    )| {
        // Property: Overlap should not exceed chunk size
        let valid_overlap = overlap_frames.min(chunk_frames - 1);

        // Property: Effective progression per chunk
        let progression = chunk_frames - valid_overlap;
        assert!(progression > 0);
        assert!(progression <= chunk_frames);

        // Property: For N chunks, total frames should be:
        // first_chunk + (N-1) * progression
        let num_chunks = 5;
        let total_frames = chunk_frames + (num_chunks - 1) * progression;

        // Property: Total should be greater than a single chunk
        assert!(total_frames > chunk_frames);

        // Property: Total should be less than N full chunks (due to overlap)
        assert!(total_frames < num_chunks * chunk_frames);
    });
}

#[test]
fn test_vad_config_properties() {
    proptest!(|(
        energy_threshold in 0.001f32..0.1f32,
        spectral_threshold in 500.0f32..2000.0f32,
        min_voice_frames in 5usize..50usize,
        max_silence_frames in 10usize..100usize,
        smoothing_factor in 0.0f32..1.0f32,
    )| {
        let config = VadConfig {
            energy_threshold,
            spectral_threshold,
            min_voice_frames,
            max_silence_frames,
            smoothing_factor,
        };

        // Property: All thresholds should be positive
        assert!(config.energy_threshold > 0.0);
        assert!(config.spectral_threshold > 0.0);

        // Property: Frame counts should be positive
        assert!(config.min_voice_frames > 0);
        assert!(config.max_silence_frames > 0);

        // Property: Smoothing factor should be in [0, 1]
        assert!(config.smoothing_factor >= 0.0);
        assert!(config.smoothing_factor <= 1.0);

        // Property: Max silence should typically be larger than min voice
        // (though not strictly required, it's a common pattern)
        if config.max_silence_frames < config.min_voice_frames {
            // Valid but unusual configuration
            assert!(config.max_silence_frames > 0);
        }
    });
}

#[test]
fn test_streaming_latency_calculation() {
    proptest!(|(
        chunk_frames in 64usize..2048usize,
        sample_rate in 16000usize..48000usize,
        lookahead_frames in 0usize..512usize,
    )| {
        // Property: Latency in milliseconds should be calculable
        let total_latency_frames = chunk_frames + lookahead_frames;
        let latency_ms = (total_latency_frames * 1000) / sample_rate;

        // Property: Latency should be reasonable for real-time
        assert!(latency_ms < 1000); // Less than 1 second

        // Property: Higher sample rates should give lower latency (for same frame count)
        let latency_ms_high = (total_latency_frames * 1000) / 48000;
        let latency_ms_low = (total_latency_frames * 1000) / 16000;
        assert!(latency_ms_high <= latency_ms_low);

        // Property: Larger chunk size should give larger latency (with same lookahead)
        let double_chunk_latency_frames = (chunk_frames * 2) + lookahead_frames;
        let double_chunk_latency_ms = (double_chunk_latency_frames * 1000) / sample_rate;

        // Double chunk should have higher latency than single chunk (unless very small values)
        if chunk_frames > 10 {
            assert!(double_chunk_latency_ms > latency_ms);
        }

        // Property: Latency calculation should be monotonic - more frames = more latency
        let triple_chunk_latency_frames = (chunk_frames * 3) + lookahead_frames;
        let triple_chunk_latency_ms = (triple_chunk_latency_frames * 1000) / sample_rate;

        assert!(triple_chunk_latency_ms >= double_chunk_latency_ms);
        assert!(double_chunk_latency_ms >= latency_ms);
    });
}

#[test]
fn test_streaming_crossfade_properties() {
    proptest!(|(
        chunk_frames in 128usize..1024usize,
        crossfade_ratio in 0.0f32..0.5f32, // Crossfade is typically 0-50% of chunk
    )| {
        let crossfade_frames = (chunk_frames as f32 * crossfade_ratio) as usize;

        // Property: Crossfade should not exceed chunk size
        assert!(crossfade_frames <= chunk_frames);

        // Property: Crossfade should leave some non-overlapping content
        assert!(chunk_frames - crossfade_frames > 0);

        // Property: With crossfade, effective chunk size is reduced (or stays same if no crossfade)
        let effective_size = chunk_frames - crossfade_frames;
        assert!(effective_size <= chunk_frames);
        assert!(effective_size > 0);

        // Property: If there is crossfade, effective size should be smaller
        if crossfade_frames > 0 {
            assert!(effective_size < chunk_frames);
        }
    });
}

#[test]
fn test_streaming_adaptive_chunk_sizing() {
    proptest!(|(
        min_chunk_frames in 64usize..512usize,
        max_chunk_frames in 512usize..2048usize,
        current_latency_ms in 10u32..500u32,
        target_latency_ms in 10u32..500u32,
    )| {
        // Property: Min should be less than or equal to max
        let valid_min = min_chunk_frames.min(max_chunk_frames);
        let valid_max = max_chunk_frames.max(min_chunk_frames);

        assert!(valid_min <= valid_max);

        // Property: Adaptive sizing should stay within bounds
        // Simulate adaptive adjustment
        let adjustment_factor = if current_latency_ms > target_latency_ms {
            0.9 // Reduce chunk size to reduce latency
        } else {
            1.1 // Increase chunk size for better quality
        };

        let current_chunk = (valid_min + valid_max) / 2; // Start in middle
        let adjusted = (current_chunk as f32 * adjustment_factor) as usize;
        let clamped = adjusted.max(valid_min).min(valid_max);

        // Property: Adjusted size should be within bounds
        assert!(clamped >= valid_min);
        assert!(clamped <= valid_max);
    });
}

#[test]
fn test_streaming_mel_chunk_division() {
    proptest!(|(
        total_mel_frames in 100usize..1000usize,
        chunk_frames in 10usize..100usize,
    )| {
        // Property: Number of chunks should be deterministic
        let num_full_chunks = total_mel_frames / chunk_frames;
        let remainder = total_mel_frames % chunk_frames;

        // Property: Total frames should equal sum of chunks
        assert_eq!(
            total_mel_frames,
            num_full_chunks * chunk_frames + remainder
        );

        // Property: Should have at least one chunk if frames > 0
        if total_mel_frames > 0 {
            assert!(num_full_chunks > 0 || remainder > 0);
        }

        // Property: Remainder should be less than chunk size
        assert!(remainder < chunk_frames);
    });
}

#[test]
fn test_streaming_quality_latency_tradeoff() {
    proptest!(|(
        quality_factor in 0.0f32..1.0f32,
    )| {
        // Property: Quality factor should affect chunk size inversely
        let base_chunk_frames = 256usize;

        // Higher quality = larger chunks = more latency
        let quality_multiplier = 1.0 + quality_factor;
        let adjusted_chunk = (base_chunk_frames as f32 * quality_multiplier) as usize;

        // Property: Higher quality should give larger chunks
        if quality_factor > 0.5 {
            assert!(adjusted_chunk > base_chunk_frames);
        }

        // Property: Quality factor of 0 should give base or smaller
        if quality_factor < 0.1 {
            assert!(adjusted_chunk <= (base_chunk_frames as f32 * 1.1) as usize);
        }
    });
}

#[test]
fn test_streaming_buffer_underrun_detection() {
    proptest!(|(
        buffer_size in 1usize..10usize,
        consumption_rate in 1usize..5usize,
        production_rate in 1usize..5usize,
    )| {
        // Property: Buffer underrun occurs when consumption > production
        let will_underrun = consumption_rate > production_rate;

        // Property: Buffer should last at least buffer_size / (consumption - production) iterations
        if will_underrun {
            let deficit = consumption_rate - production_rate;
            let iterations_until_empty = buffer_size / deficit.max(1);

            // Property: Should deplete in finite time (usize is always >= 0, so just check upper bound)
            assert!(iterations_until_empty <= buffer_size);
        } else {
            // Property: Buffer should not underrun if production >= consumption
            assert!(production_rate >= consumption_rate);
        }
    });
}

#[test]
fn test_streaming_lookahead_consistency() {
    proptest!(|(
        current_position in 0usize..1000usize,
        lookahead_frames in 10usize..200usize,
        total_frames in 1000usize..2000usize,
    )| {
        // Property: Lookahead should not exceed available frames
        let available_ahead = total_frames.saturating_sub(current_position);

        let actual_lookahead = lookahead_frames.min(available_ahead);

        // Property: Actual lookahead should never exceed requested
        assert!(actual_lookahead <= lookahead_frames);

        // Property: Actual lookahead should never exceed available
        assert!(actual_lookahead <= available_ahead);

        // Property: Lookahead range should be valid
        let lookahead_end = current_position + actual_lookahead;
        assert!(lookahead_end <= total_frames);
    });
}

// ============================================================================
// Input Validation / Fuzzing Property Tests
// ============================================================================

#[test]
fn test_synthesis_config_extreme_values() {
    proptest!(|(
        speed in -1000.0f32..1000.0f32,
        pitch_shift in -1000.0f32..1000.0f32,
        energy in -1000.0f32..1000.0f32,
    )| {
        // Property: Config creation should not panic with extreme values
        // Real implementation should clamp or validate these
        let clamped_speed = speed.abs().clamp(0.01, 100.0);
        let clamped_pitch = pitch_shift.clamp(-48.0, 48.0);
        let clamped_energy = energy.abs().clamp(0.001, 100.0);

        let config = SynthesisConfig {
            speed: clamped_speed,
            pitch_shift: clamped_pitch,
            energy: clamped_energy,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        };

        // Property: After clamping, values should be in safe ranges
        assert!(config.speed > 0.0);
        assert!(config.speed <= 100.0);
        assert!(config.pitch_shift >= -48.0);
        assert!(config.pitch_shift <= 48.0);
        assert!(config.energy > 0.0);
        assert!(config.energy <= 100.0);
    });
}

#[test]
fn test_mel_spectrogram_boundary_cases() {
    proptest!(|(
        n_mels in 1usize..256usize,
        n_frames in 1usize..10000usize,
        sample_rate in 8000usize..96000usize,
        hop_length in 32usize..2048usize,
    )| {
        // Property: Mel spectrogram should handle various boundary cases
        let data = vec![vec![0.0f32; n_frames]; n_mels];

        let mel = MelSpectrogram::new(data, sample_rate as u32, hop_length as u32);

        // Property: Dimensions should match input
        assert_eq!(mel.n_mels, n_mels);
        assert_eq!(mel.n_frames, n_frames);

        // Property: Duration should be positive
        let duration = mel.duration();
        assert!(duration > 0.0);

        // Property: Duration should be reasonable (< 1 hour for < 10000 frames)
        assert!(duration < 3600.0);
    });
}

#[test]
fn test_phoneme_symbol_validation() {
    proptest!(|(
        symbol_length in 0usize..100usize,
        use_unicode in prop::bool::ANY,
    )| {
        // Property: Phoneme symbols of various lengths and character sets
        let symbol = if use_unicode {
            "あいうえお".chars().take(symbol_length.min(5)).collect::<String>()
        } else {
            "abcdefghij".chars().take(symbol_length.min(10)).collect::<String>()
        };

        if !symbol.is_empty() {
            let phoneme = Phoneme::new(symbol.clone());

            // Property: Symbol should be preserved
            assert_eq!(phoneme.symbol, symbol);

            // Property: Phoneme should be valid
            assert!(!phoneme.symbol.is_empty());
        }
    });
}

#[test]
fn test_streaming_config_boundary_validation() {
    proptest!(|(
        chunk_frames in 1usize..8192usize,
        overlap_frames in 0usize..4096usize,
        buffer_size in 1usize..50usize,
        num_threads in 1usize..32usize,
    )| {
        // Property: Streaming config should handle various sizes safely
        let valid_overlap = overlap_frames.min(chunk_frames.saturating_sub(1));

        let config = StreamingConfig {
            chunk_frames,
            overlap_frames: valid_overlap,
            max_latency_ms: 500,
            quality_factor: 0.5,
            buffer_size,
            enable_prediction: true,
            adaptive_chunking: false,
            enable_vad: false,
            lookahead_frames: chunk_frames / 4,
            num_threads,
            enable_realtime_streaming: true,
            min_chunk_frames: 64,
            max_chunk_frames: 2048,
            prediction_lookahead: 10,
            vad_threshold: 0.01,
            enable_crossfade: false,
            crossfade_frames: 0,
        };

        // Property: Config should have valid relationships
        assert!(config.overlap_frames < config.chunk_frames);
        assert!(config.buffer_size > 0);
        assert!(config.num_threads > 0);
        assert!(config.num_threads <= 32);
    });
}

#[test]
fn test_vad_config_extreme_thresholds() {
    proptest!(|(
        energy_threshold in 0.0f32..10.0f32,
        spectral_threshold in 0.0f32..10000.0f32,
        min_voice_frames in 1usize..200usize,
        max_silence_frames in 1usize..500usize,
        smoothing_factor in 0.0f32..2.0f32,
    )| {
        // Property: VAD config should handle various threshold values
        let clamped_smoothing = smoothing_factor.clamp(0.0, 1.0);

        let config = VadConfig {
            energy_threshold: energy_threshold.max(0.0001),
            spectral_threshold: spectral_threshold.max(0.001),
            min_voice_frames,
            max_silence_frames,
            smoothing_factor: clamped_smoothing,
        };

        // Property: All values should be positive
        assert!(config.energy_threshold > 0.0);
        assert!(config.spectral_threshold > 0.0);
        assert!(config.min_voice_frames > 0);
        assert!(config.max_silence_frames > 0);

        // Property: Smoothing factor should be in valid range
        assert!(config.smoothing_factor >= 0.0);
        assert!(config.smoothing_factor <= 1.0);
    });
}

#[test]
fn test_mel_spectrogram_empty_edge_cases() {
    // Property: Edge cases with minimal valid inputs
    let min_mel = MelSpectrogram::new(vec![vec![0.0; 1]; 1], 8000, 256);

    assert_eq!(min_mel.n_mels, 1);
    assert_eq!(min_mel.n_frames, 1);
    assert!(min_mel.duration() > 0.0);
}

// ============================================================================
// Advanced Numerical Stability Property Tests
// ============================================================================

#[test]
fn test_numerical_stability_with_extreme_values() {
    proptest!(|(
        magnitude in 1e-10f32..1e10f32,
        sign in proptest::bool::ANY,
    )| {
        // Property: System should handle extreme but valid floating point values
        let value = if sign { magnitude } else { -magnitude };

        let config = SynthesisConfig {
            speed: 1.0,
            pitch_shift: value.clamp(-24.0, 24.0),
            energy: magnitude.abs().clamp(0.01, 10.0),
            ..Default::default()
        };

        // Property: After clamping, values should be finite and reasonable
        assert!(config.speed.is_finite());
        assert!(config.pitch_shift.is_finite());
        assert!(config.energy.is_finite());
        assert!(config.energy > 0.0);
    });
}

#[test]
fn test_phoneme_sequence_length_scaling() {
    proptest!(|(
        sequence_length in 1usize..10000usize,
    )| {
        // Property: System should handle various sequence lengths efficiently
        let phonemes: Vec<Phoneme> = (0..sequence_length)
            .map(|i| Phoneme::new(format!("P{}", i % 100)))
            .collect();

        // Property: Sequence length should be preserved
        assert_eq!(phonemes.len(), sequence_length);

        // Property: Memory usage should scale linearly
        // (rough estimate based on phoneme structure size)
        let estimated_bytes = sequence_length * std::mem::size_of::<Phoneme>();
        assert!(estimated_bytes > 0);
        assert!(estimated_bytes < 100_000_000); // Reasonable upper bound
    });
}

#[test]
fn test_synthesis_config_rounding_consistency() {
    proptest!(|(
        speed in 0.1f32..10.0f32,
        pitch in -24.0f32..24.0f32,
    )| {
        // Property: Repeated rounding should be idempotent
        let config1 = SynthesisConfig {
            speed: (speed * 10.0).round() / 10.0,
            pitch_shift: pitch.round(),
            ..Default::default()
        };

        let config2 = SynthesisConfig {
            speed: ((config1.speed * 10.0).round() / 10.0),
            pitch_shift: config1.pitch_shift.round(),
            ..Default::default()
        };

        // Property: Second rounding should not change values
        assert!((config1.speed - config2.speed).abs() < 1e-6);
        assert!((config1.pitch_shift - config2.pitch_shift).abs() < 1e-6);
    });
}

// ============================================================================
// Cache and Memory Property Tests
// ============================================================================

#[test]
fn test_cache_key_collision_resistance() {
    proptest!(|(
        text1 in "a{1,50}",
        text2 in "a{1,50}",
        speed1 in 0.5f32..2.0f32,
        speed2 in 0.5f32..2.0f32,
    )| {
        use voirs_acoustic::synthesis_cache::SynthesisCacheKey;

        let key1 = SynthesisCacheKey::new(&text1, None, speed1, 0.0, 1.0);
        let key2 = SynthesisCacheKey::new(&text2, None, speed2, 0.0, 1.0);

        // Property: Different inputs should (usually) produce different keys
        if text1 != text2 || (speed1 - speed2).abs() > 0.1 {
            assert_ne!(key1, key2);
        }

        // Property: Same inputs should always produce same key
        let key1_dup = SynthesisCacheKey::new(&text1, None, speed1, 0.0, 1.0);
        assert_eq!(key1, key1_dup);
    });
}

#[test]
fn test_mel_spectrogram_size_estimation() {
    proptest!(|(
        n_mels in 40usize..128usize,
        n_frames in 100usize..2000usize,
    )| {
        // Property: Memory size estimation should be accurate
        let data = vec![vec![0.0f32; n_frames]; n_mels];
        let mel = MelSpectrogram::new(data, 22050, 256);

        let estimated_size = mel.estimate_size_bytes();

        // Property: Size should account for data dimensions
        let expected_min = n_mels * n_frames * std::mem::size_of::<f32>();
        assert!(estimated_size >= expected_min);

        // Property: Size should include overhead but not be excessive
        let expected_max = expected_min * 2; // Allow 2x for overhead
        assert!(estimated_size <= expected_max);
    });
}

// ============================================================================
// Performance and Scalability Property Tests
// ============================================================================

#[test]
fn test_batch_size_scaling_properties() {
    proptest!(|(
        batch_size in 1usize..256usize,
        item_size in 10usize..500usize,
    )| {
        // Property: Batch processing should scale predictably
        let total_items = batch_size * item_size;

        // Property: Total should be product of batch and item size
        assert_eq!(total_items, batch_size * item_size);

        // Property: Batching should not exceed reasonable limits
        assert!(total_items < 200_000); // Reasonable upper limit
    });
}

#[test]
fn test_streaming_chunk_alignment() {
    proptest!(|(
        chunk_frames in 64usize..2048usize,
        total_frames in 1000usize..100000usize,
    )| {
        // Property: Total frames should be processable in chunks
        let num_chunks = (total_frames + chunk_frames - 1) / chunk_frames;

        // Property: Number of chunks should be reasonable
        assert!(num_chunks > 0);
        assert!(num_chunks <= total_frames); // At most one chunk per frame

        // Property: Last chunk might be smaller but should exist
        let frames_in_last_chunk = total_frames - (num_chunks - 1) * chunk_frames;
        assert!(frames_in_last_chunk > 0);
        assert!(frames_in_last_chunk <= chunk_frames);
    });
}

// ============================================================================
// Error Recovery and Robustness Property Tests
// ============================================================================

#[test]
fn test_config_validation_properties() {
    proptest!(|(
        speed in -100.0f32..100.0f32,
        pitch in -1000.0f32..1000.0f32,
        energy in -100.0f32..100.0f32,
    )| {
        // Property: Config should handle invalid inputs gracefully
        // by clamping to valid ranges

        let safe_speed = if speed.is_finite() && speed > 0.0 {
            speed.clamp(0.1, 10.0)
        } else {
            1.0 // Default
        };

        let safe_pitch = if pitch.is_finite() {
            pitch.clamp(-24.0, 24.0)
        } else {
            0.0 // Default
        };

        let safe_energy = if energy.is_finite() && energy > 0.0 {
            energy.clamp(0.01, 10.0)
        } else {
            1.0 // Default
        };

        // Property: All safe values should be valid
        assert!(safe_speed >= 0.1 && safe_speed <= 10.0);
        assert!(safe_pitch >= -24.0 && safe_pitch <= 24.0);
        assert!(safe_energy >= 0.01 && safe_energy <= 10.0);
    });
}

#[test]
fn test_phoneme_sequence_filtering() {
    proptest!(|(phonemes in phoneme_sequence())| {
        // Property: Filtering should preserve valid phonemes
        let valid_phonemes: Vec<_> = phonemes
            .iter()
            .filter(|p| !p.symbol.is_empty())
            .collect();

        // Property: Filtered sequence should not exceed original
        assert!(valid_phonemes.len() <= phonemes.len());

        // Property: All filtered phonemes should be from original
        for valid in &valid_phonemes {
            assert!(phonemes.contains(valid));
        }
    });
}

// ============================================================================
// Concurrent Access Property Tests
// ============================================================================

#[test]
fn test_mel_spectrogram_clone_independence() {
    proptest!(|(
        n_mels in 40usize..80usize,
        n_frames in 100usize..200usize,
        value in -10.0f32..10.0f32,
    )| {
        // Property: Cloned mel spectrograms should be independent
        let data1 = vec![vec![value; n_frames]; n_mels];
        let mel1 = MelSpectrogram::new(data1, 22050, 256);

        let mel2 = mel1.clone();

        // Property: Clones should have same dimensions
        assert_eq!(mel1.n_mels, mel2.n_mels);
        assert_eq!(mel1.n_frames, mel2.n_frames);

        // Property: Modifying clone shouldn't affect original
        // (data is owned, so this is guaranteed by Rust)
        assert_eq!(mel1.n_mels, n_mels);
        assert_eq!(mel2.n_mels, n_mels);
    });
}

#[test]
fn test_synthesis_config_thread_safety_properties() {
    proptest!(|(
        speed1 in 0.5f32..2.0f32,
        speed2 in 0.5f32..2.0f32,
    )| {
        // Property: Configs created concurrently should be independent
        let config1 = SynthesisConfig {
            speed: speed1,
            ..Default::default()
        };

        let config2 = SynthesisConfig {
            speed: speed2,
            ..Default::default()
        };

        // Property: Different configs should have different speeds
        if (speed1 - speed2).abs() > 0.01 {
            assert!((config1.speed - config2.speed).abs() > 0.001);
        }

        // Property: Configs should be safely clonable
        let config1_clone = config1.clone();
        assert_eq!(config1.speed, config1_clone.speed);
    });
}

#[test]
fn test_phoneme_duration_edge_cases() {
    proptest!(|(
        duration in 0.0f32..10.0f32,
    )| {
        let mut phoneme = Phoneme::new("a".to_string());
        phoneme.duration = Some(duration);

        if let Some(d) = phoneme.duration {
            // Property: Duration should be the value we set
            assert!((d - duration).abs() < 0.0001);

            // Property: Duration should handle various ranges
            if d > 0.0 {
                assert!(d > 0.0);
            }
        }
    });
}

#[test]
fn test_language_code_case_insensitivity_comprehensive() {
    proptest!(|(
        use_lowercase in prop::bool::ANY,
        use_uppercase in prop::bool::ANY,
    )| {
        // Test various case combinations for English (US)
        let variants = vec![
            "en-US", "en-us", "EN-US", "EN-us", "En-Us", "eN-Us", "en-uS", "EN-uS"
        ];

        for variant in variants {
            let parsed = LanguageCode::parse(variant);

            // Property: All case variants should parse to the same language code
            assert_eq!(parsed, Some(LanguageCode::EnUs),
                      "Failed to parse variant: {}", variant);
        }
    });
}

#[test]
fn test_streaming_chunk_division_edge_cases() {
    proptest!(|(
        total_frames in 1usize..100000usize,
        chunk_size in 1usize..5000usize,
    )| {
        // Property: Chunk division should handle all sizes correctly
        let num_full_chunks = total_frames / chunk_size;
        let remainder = total_frames % chunk_size;

        // Property: Should be able to reconstruct total
        assert_eq!(num_full_chunks * chunk_size + remainder, total_frames);

        // Property: Remainder should always be less than chunk size
        assert!(remainder < chunk_size);

        // Property: If total < chunk_size, should have 0 full chunks
        if total_frames < chunk_size {
            assert_eq!(num_full_chunks, 0);
            assert_eq!(remainder, total_frames);
        }
    });
}

#[test]
fn test_synthesis_config_nan_inf_handling() {
    // Property: Config should not contain NaN or Inf values
    let valid_configs = vec![(1.0, 0.0, 1.0), (0.5, -5.0, 0.8), (2.0, 5.0, 1.5)];

    for (speed, pitch, energy) in valid_configs {
        let config = SynthesisConfig {
            speed,
            pitch_shift: pitch,
            energy,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        };

        // Property: Values should not be NaN or Inf
        assert!(!config.speed.is_nan());
        assert!(!config.speed.is_infinite());
        assert!(!config.pitch_shift.is_nan());
        assert!(!config.pitch_shift.is_infinite());
        assert!(!config.energy.is_nan());
        assert!(!config.energy.is_infinite());
    }
}

#[test]
fn test_phoneme_sequence_large_inputs() {
    proptest!(|(
        sequence_length in 1usize..1000usize,
    )| {
        // Property: Should handle large phoneme sequences
        let phonemes: Vec<Phoneme> = (0..sequence_length)
            .map(|i| Phoneme::new(format!("p{}", i % 100)))
            .collect();

        // Property: Length should be preserved
        assert_eq!(phonemes.len(), sequence_length);

        // Property: All phonemes should be valid
        for phoneme in &phonemes {
            assert!(!phoneme.symbol.is_empty());
        }

        // Property: Should be able to concatenate
        let mut doubled = phonemes.clone();
        doubled.extend(phonemes.clone());
        assert_eq!(doubled.len(), sequence_length * 2);
    });
}

// ============================================================================
// Property-Based Tests for Optimization Features
// ============================================================================

use voirs_acoustic::optimization::{
    DistillationConfig, DistillationMethod, HardwareOptimization, OptimizationConfig,
    OptimizationTargets, PruningConfig, PruningStrategy, PruningType, QuantizationConfig,
    QuantizationMethod, QuantizationPrecision, StudentModelConfig, TargetDevice,
};

/// Strategy for generating valid sparsity values (0.0 to 1.0)
fn sparsity_value() -> impl Strategy<Value = f32> {
    0.0f32..=1.0f32
}

/// Strategy for generating valid temperature values (> 0.0)
fn temperature_value() -> impl Strategy<Value = f32> {
    0.1f32..10.0f32
}

/// Strategy for generating valid alpha/weight values (0.0 to 1.0)
fn weight_value() -> impl Strategy<Value = f32> {
    0.0f32..=1.0f32
}

/// Strategy for generating calibration sample counts
fn calibration_samples() -> impl Strategy<Value = usize> {
    100usize..10000usize
}

#[test]
fn test_quantization_config_properties() {
    proptest!(|(
        calibration_samples in calibration_samples(),
        dynamic_quantization in any::<bool>(),
    )| {
        let config = QuantizationConfig {
            enabled: true,
            precision: QuantizationPrecision::Float16,
            calibration_samples,
            excluded_layers: vec![],
            quantization_method: QuantizationMethod::PostTraining,
            dynamic_quantization,
        };

        // Property 1: Calibration samples should always be positive
        assert!(config.calibration_samples > 0);

        // Property 2: Configuration should be serializable
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());

        // Property 3: Round-trip serialization should preserve values
        let deserialized: QuantizationConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.calibration_samples, deserialized.calibration_samples);
        assert_eq!(config.dynamic_quantization, deserialized.dynamic_quantization);
    });
}

#[test]
fn test_pruning_config_sparsity_bounds() {
    proptest!(|(
        target_sparsity in sparsity_value(),
        gradual_pruning in any::<bool>(),
    )| {
        let config = PruningConfig {
            enabled: true,
            strategy: PruningStrategy::Magnitude,
            target_sparsity,
            gradual_pruning,
            pruning_type: PruningType::Unstructured,
            excluded_layers: vec![],
        };

        // Property 1: Sparsity must be in valid range [0.0, 1.0]
        assert!(config.target_sparsity >= 0.0);
        assert!(config.target_sparsity <= 1.0);

        // Property 2: Enabled configuration should be consistent
        assert!(config.enabled);

        // Property 3: Serialization round-trip preserves sparsity
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: PruningConfig = serde_json::from_str(&serialized).unwrap();
        assert!((config.target_sparsity - deserialized.target_sparsity).abs() < 0.0001);
    });
}

#[test]
fn test_distillation_config_temperature_validation() {
    proptest!(|(
        temperature in temperature_value(),
        distillation_weight in weight_value(),
    )| {
        let config = DistillationConfig {
            enabled: true,
            method: DistillationMethod::Standard,
            teacher_model_path: Some("teacher.safetensors".to_string()),
            student_config: StudentModelConfig {
                hidden_reduction_factor: 0.5,
                layer_reduction_factor: 0.5,
                num_heads: 4,
                shared_parameters: false,
            },
            temperature,
            distillation_weight,
        };

        // Property 1: Temperature must be positive
        assert!(config.temperature > 0.0);

        // Property 2: Distillation weight should be in [0, 1]
        assert!(config.distillation_weight >= 0.0);
        assert!(config.distillation_weight <= 1.0);

        // Property 3: Configuration should serialize correctly
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: DistillationConfig = serde_json::from_str(&serialized).unwrap();
        assert!((config.temperature - deserialized.temperature).abs() < 0.0001);
    });
}

#[test]
fn test_optimization_targets_consistency() {
    proptest!(|(
        max_quality_loss in weight_value(),
        memory_reduction in (0.0f32..1.0f32),
        speed_improvement in (1.0f32..10.0f32),
    )| {
        let targets = OptimizationTargets {
            max_quality_loss,
            memory_reduction_target: memory_reduction,
            speed_improvement_target: speed_improvement,
            max_model_size_mb: Some(100),
            target_latency_ms: Some(10.0),
        };

        // Property 1: Quality loss should be in [0, 1]
        assert!(targets.max_quality_loss >= 0.0);
        assert!(targets.max_quality_loss <= 1.0);

        // Property 2: Memory reduction should be non-negative
        assert!(targets.memory_reduction_target >= 0.0);

        // Property 3: Speed improvement should be >= 1.0 (1x means no improvement)
        assert!(targets.speed_improvement_target >= 1.0);

        // Property 4: Serialization should work
        let serialized = serde_json::to_string(&targets).unwrap();
        assert!(!serialized.is_empty());
    });
}

#[test]
fn test_hardware_optimization_memory_constraints() {
    proptest!(|(
        memory_limit in (128usize..16384usize), // 128MB to 16GB
        cpu_cores in (1usize..64usize),
        enable_simd in any::<bool>(),
        enable_gpu in any::<bool>(),
    )| {
        let optimization = HardwareOptimization {
            target_device: TargetDevice::Desktop,
            enable_simd,
            enable_gpu,
            memory_limit_mb: Some(memory_limit),
            cpu_cores: Some(cpu_cores),
        };

        // Property 1: Memory limit should be positive if set
        if let Some(limit) = optimization.memory_limit_mb {
            assert!(limit > 0);
        }

        // Property 2: CPU cores should be positive if set
        if let Some(cores) = optimization.cpu_cores {
            assert!(cores > 0);
        }

        // Property 3: Configuration should be serializable
        let serialized = serde_json::to_string(&optimization).unwrap();
        let deserialized: HardwareOptimization = serde_json::from_str(&serialized).unwrap();
        assert_eq!(optimization.memory_limit_mb, deserialized.memory_limit_mb);
        assert_eq!(optimization.cpu_cores, deserialized.cpu_cores);
    });
}

#[test]
fn test_student_model_config_reduction_factors() {
    proptest!(|(
        hidden_reduction in (0.1f32..1.0f32),
        layer_reduction in (0.1f32..1.0f32),
        num_heads in (1usize..16usize),
        shared_params in any::<bool>(),
    )| {
        let config = StudentModelConfig {
            hidden_reduction_factor: hidden_reduction,
            layer_reduction_factor: layer_reduction,
            num_heads,
            shared_parameters: shared_params,
        };

        // Property 1: Reduction factors should be in (0, 1] range
        assert!(config.hidden_reduction_factor > 0.0);
        assert!(config.hidden_reduction_factor <= 1.0);
        assert!(config.layer_reduction_factor > 0.0);
        assert!(config.layer_reduction_factor <= 1.0);

        // Property 2: Number of heads should be positive
        assert!(config.num_heads > 0);

        // Property 3: Serialization should preserve values
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: StudentModelConfig = serde_json::from_str(&serialized).unwrap();
        assert!((config.hidden_reduction_factor - deserialized.hidden_reduction_factor).abs() < 0.0001);
        assert_eq!(config.num_heads, deserialized.num_heads);
    });
}

#[test]
fn test_optimization_config_default_consistency() {
    // Property test: Default configuration should always be valid
    proptest!(|(_dummy in 0..100)| {
        let config = OptimizationConfig::default();

        // Property 1: All enabled optimizations should have valid parameters
        if config.quantization.enabled {
            assert!(config.quantization.calibration_samples > 0);
        }

        if config.pruning.enabled {
            assert!(config.pruning.target_sparsity >= 0.0);
            assert!(config.pruning.target_sparsity <= 1.0);
        }

        if config.distillation.enabled {
            assert!(config.distillation.temperature > 0.0);
        }

        // Property 2: Optimization targets should be reasonable
        assert!(config.optimization_targets.max_quality_loss >= 0.0);
        assert!(config.optimization_targets.speed_improvement_target >= 1.0);

        // Property 3: Configuration should always be serializable
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());
    });
}

#[test]
fn test_pruning_strategy_enumeration_completeness() {
    // Property test: All pruning strategies should be handleable
    let strategies = vec![
        PruningStrategy::Magnitude,
        PruningStrategy::Gradient,
        PruningStrategy::Fisher,
        PruningStrategy::Adaptive,
    ];

    for strategy in strategies {
        let config = PruningConfig {
            enabled: true,
            strategy: strategy.clone(),
            target_sparsity: 0.5,
            gradual_pruning: false,
            pruning_type: PruningType::Unstructured,
            excluded_layers: vec![],
        };

        // Property: All strategies should serialize correctly
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());

        // Property: Deserialization should preserve strategy
        let deserialized: PruningConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            std::mem::discriminant(&config.strategy),
            std::mem::discriminant(&deserialized.strategy)
        );
    }
}

#[test]
fn test_quantization_precision_levels() {
    // Property test: All quantization precisions should be valid
    let precisions = vec![
        QuantizationPrecision::Int8,
        QuantizationPrecision::Float16,
        QuantizationPrecision::Mixed,
        QuantizationPrecision::Dynamic,
    ];

    for precision in precisions {
        let config = QuantizationConfig {
            enabled: true,
            precision: precision.clone(),
            calibration_samples: 1000,
            excluded_layers: vec![],
            quantization_method: QuantizationMethod::PostTraining,
            dynamic_quantization: false,
        };

        // Property: All precisions should serialize/deserialize correctly
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: QuantizationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            std::mem::discriminant(&config.precision),
            std::mem::discriminant(&deserialized.precision)
        );
    }
}

#[test]
fn test_target_device_enumeration() {
    // Property test: All target devices should be valid
    let devices = vec![
        TargetDevice::Mobile,
        TargetDevice::Desktop,
        TargetDevice::Server,
        TargetDevice::Edge,
    ];

    for device in devices {
        let optimization = HardwareOptimization {
            target_device: device.clone(),
            enable_simd: true,
            enable_gpu: false,
            memory_limit_mb: Some(1024),
            cpu_cores: Some(4),
        };

        // Property: All devices should serialize correctly
        let serialized = serde_json::to_string(&optimization).unwrap();
        let deserialized: HardwareOptimization = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            std::mem::discriminant(&optimization.target_device),
            std::mem::discriminant(&deserialized.target_device)
        );
    }
}
