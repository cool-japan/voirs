//! SciRS2 Integration Tests
//!
//! Real-world integration tests demonstrating practical usage of SciRS2-optimized
//! operations in acoustic processing pipelines.

use std::sync::Arc;
use voirs_acoustic::scirs2_ops::{
    NormalizationMethod, SciRS2MelOps, SciRS2NumericOps, SciRS2ParallelOps,
};
use voirs_acoustic::MelSpectrogram;

fn create_realistic_mel(n_mels: usize, n_frames: usize) -> MelSpectrogram {
    // Simulate realistic mel spectrogram data from actual speech
    let mut data = Vec::with_capacity(n_mels);

    for mel_idx in 0..n_mels {
        let mut channel = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            // Simulate speech-like energy distribution
            // Lower frequencies have more energy
            let freq_factor = 1.0 - (mel_idx as f32 / n_mels as f32).powi(2);

            // Time-varying envelope (simulates prosody)
            let time_factor = (frame_idx as f32 / n_frames as f32 * std::f32::consts::PI).sin();

            // Add some noise
            let noise = (fastrand::f32() - 0.5) * 0.1;

            let value = freq_factor * (0.5 + 0.5 * time_factor) + noise;
            channel.push(value.max(0.0)); // Mel values are non-negative
        }

        data.push(channel);
    }

    MelSpectrogram {
        data,
        sample_rate: 16000,
        hop_length: 256,
        n_mels,
        n_frames,
    }
}

#[test]
fn test_complete_preprocessing_pipeline() {
    // Simulate a complete preprocessing pipeline for TTS training

    let batch_size = 8;
    let n_mels = 80;
    let n_frames = 500;

    // 1. Create batch of mel spectrograms
    let mut mels: Vec<MelSpectrogram> = (0..batch_size)
        .map(|_| create_realistic_mel(n_mels, n_frames))
        .collect();

    // 2. Apply parallel normalization
    SciRS2MelOps::batch_normalize_parallel(&mut mels, NormalizationMethod::ZScore).unwrap();

    // 3. Verify all mels are properly normalized
    for mel in &mels {
        let all_values: Vec<f32> = mel.data.iter().flatten().copied().collect();
        let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
        let std = (all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / all_values.len() as f32)
            .sqrt();

        assert!(
            (mean.abs()) < 0.01,
            "Mean should be close to 0, got {}",
            mean
        );
        assert!(
            (std - 1.0).abs() < 0.01,
            "Std should be close to 1, got {}",
            std
        );
    }

    // 4. Convert to ndarray for advanced processing
    let arrays: Vec<_> = mels
        .iter()
        .map(|mel| SciRS2MelOps::to_ndarray(mel).unwrap())
        .collect();

    assert_eq!(arrays.len(), batch_size);
    assert_eq!(arrays[0].shape(), &[n_mels, n_frames]);

    println!("✓ Complete preprocessing pipeline test passed");
}

#[test]
fn test_parallel_phoneme_encoding_workflow() {
    // Simulate phoneme encoding for multiple utterances

    let phoneme_sequences = vec![
        vec![
            "HH".to_string(),
            "EH".to_string(),
            "L".to_string(),
            "OW".to_string(),
        ], // "hello"
        vec![
            "W".to_string(),
            "ER".to_string(),
            "L".to_string(),
            "D".to_string(),
        ], // "world"
        vec![
            "T".to_string(),
            "EH".to_string(),
            "S".to_string(),
            "T".to_string(),
        ], // "test"
    ];

    // Mock phoneme encoder (in real usage, this would be a neural network)
    let encoder = |phonemes: &[String]| -> Vec<f32> {
        // Convert phonemes to embedding vectors
        phonemes
            .iter()
            .flat_map(|p| {
                // Simple hash-based embedding (256-dim)
                let hash = p
                    .bytes()
                    .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
                (0..256)
                    .map(|i| ((hash.wrapping_mul(i) % 1000) as f32) / 1000.0)
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    // Parallel encoding
    let encodings = SciRS2ParallelOps::parallel_phoneme_encoding(&phoneme_sequences, encoder);

    assert_eq!(encodings.len(), 3);
    assert_eq!(encodings[0].len(), 4 * 256); // 4 phonemes × 256-dim embedding
    assert_eq!(encodings[1].len(), 4 * 256);
    assert_eq!(encodings[2].len(), 4 * 256);

    println!("✓ Parallel phoneme encoding workflow test passed");
}

#[test]
fn test_fft_based_processing_pipeline() {
    // Simulate FFT-based acoustic feature extraction

    let n_fft = 1024;
    let n_frames = 100;

    // Generate simulated FFT output (complex spectrogram)
    let mut real_parts = Vec::new();
    let mut imag_parts = Vec::new();

    for _ in 0..n_frames {
        let real: Vec<f32> = (0..n_fft / 2).map(|i| (i as f32 * 0.01).sin()).collect();
        let imag: Vec<f32> = (0..n_fft / 2).map(|i| (i as f32 * 0.01).cos()).collect();
        real_parts.push(real);
        imag_parts.push(imag);
    }

    // Process each frame
    let mut magnitude_spectrograms = Vec::new();
    let mut phase_spectrograms = Vec::new();

    for (real, imag) in real_parts.iter().zip(imag_parts.iter()) {
        let complex = SciRS2NumericOps::complex_mel_transform(real, imag);
        let magnitudes = SciRS2NumericOps::compute_magnitude_spectrum(&complex);
        let phases = SciRS2NumericOps::compute_phase_spectrum(&complex);

        magnitude_spectrograms.push(magnitudes);
        phase_spectrograms.push(phases);
    }

    assert_eq!(magnitude_spectrograms.len(), n_frames);
    assert_eq!(phase_spectrograms.len(), n_frames);
    assert_eq!(magnitude_spectrograms[0].len(), n_fft / 2);

    // Verify magnitude is always non-negative
    for magnitudes in &magnitude_spectrograms {
        assert!(
            magnitudes.iter().all(|&m| m >= 0.0),
            "Magnitudes must be non-negative"
        );
    }

    // Verify phase is in [-π, π]
    for phases in &phase_spectrograms {
        assert!(
            phases
                .iter()
                .all(|&p| p >= -std::f32::consts::PI && p <= std::f32::consts::PI),
            "Phases must be in [-π, π]"
        );
    }

    println!("✓ FFT-based processing pipeline test passed");
}

#[test]
fn test_multi_speaker_batch_processing() {
    // Simulate processing for multiple speakers in parallel

    let num_speakers = 4;
    let utterances_per_speaker = 3;
    let n_mels = 80;
    let n_frames = 200;

    // Create speaker-specific mel spectrograms
    let mut all_mels = Vec::new();
    for speaker_id in 0..num_speakers {
        for utterance_id in 0..utterances_per_speaker {
            let mut mel = create_realistic_mel(n_mels, n_frames);

            // Add speaker-specific characteristics (simulate speaker embedding effect)
            let speaker_bias = (speaker_id as f32) * 0.1;
            for channel in &mut mel.data {
                for val in channel.iter_mut() {
                    *val += speaker_bias;
                }
            }

            all_mels.push(mel);
        }
    }

    assert_eq!(all_mels.len(), num_speakers * utterances_per_speaker);

    // Normalize all mels in parallel
    SciRS2MelOps::batch_normalize_parallel(&mut all_mels, NormalizationMethod::MinMax).unwrap();

    // Verify normalization
    for mel in &all_mels {
        for channel in &mel.data {
            let min = channel.iter().copied().fold(f32::INFINITY, f32::min);
            let max = channel.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            assert!(
                min >= 0.0 - 1e-6,
                "Min should be >= 0 after min-max normalization"
            );
            assert!(
                max <= 1.0 + 1e-6,
                "Max should be <= 1 after min-max normalization"
            );
        }
    }

    println!("✓ Multi-speaker batch processing test passed");
}

#[test]
fn test_streaming_synthesis_simulation() {
    // Simulate streaming TTS synthesis with chunked processing

    let chunk_size = 50; // frames per chunk
    let total_chunks = 10;
    let n_mels = 80;

    // Mock synthesis function that generates mel chunks
    let synthesize_chunk = Arc::new(|text: &str| -> Vec<f32> {
        // Generate chunk of mel spectrogram data
        (0..chunk_size * n_mels)
            .map(|i| {
                let text_hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                ((text_hash.wrapping_mul(i as u32) % 1000) as f32) / 1000.0
            })
            .collect()
    });

    // Generate text chunks
    let text_chunks: Vec<String> = (0..total_chunks).map(|i| format!("chunk_{}", i)).collect();

    // Process chunks in parallel
    let mel_chunks = SciRS2ParallelOps::parallel_synthesis(&text_chunks, synthesize_chunk);

    assert_eq!(mel_chunks.len(), total_chunks);
    assert_eq!(mel_chunks[0].len(), chunk_size * n_mels);

    // Verify all chunks are generated
    assert!(mel_chunks
        .iter()
        .all(|chunk| chunk.len() == chunk_size * n_mels));

    println!("✓ Streaming synthesis simulation test passed");
}

#[test]
fn test_numerical_stability_edge_cases() {
    // Test numerical stability with extreme values

    // Very small values
    let small_values = vec![1e-10, 1e-8, 1e-6, 1e-4];
    let log_small = SciRS2NumericOps::compute_log_mel(&small_values, 1e-12);
    assert!(
        log_small.iter().all(|&x| x.is_finite()),
        "Log of small values should be finite"
    );

    // Very large values
    let large_values = vec![1e4, 1e6, 1e8];
    let log_large = SciRS2NumericOps::compute_log_mel(&large_values, 1e-12);
    assert!(
        log_large.iter().all(|&x| x.is_finite()),
        "Log of large values should be finite"
    );

    // Division by near-zero
    let numerator = vec![1.0, 2.0, 3.0];
    let near_zero_denom = vec![1e-12, 1e-10, 1e-8];
    let division_result = SciRS2NumericOps::safe_divide(&numerator, &near_zero_denom, 1e-10);
    assert!(
        division_result.iter().all(|&x| x.is_finite()),
        "Division results should be finite"
    );

    // Mixed scale division
    let mixed_num = vec![1e-6, 1.0, 1e6];
    let mixed_denom = vec![1e-6, 1.0, 1e6];
    let mixed_result = SciRS2NumericOps::safe_divide(&mixed_num, &mixed_denom, 1e-10);
    assert!(
        mixed_result.iter().all(|&x| x.is_finite()),
        "Mixed scale results should be finite"
    );

    println!("✓ Numerical stability edge cases test passed");
}

#[test]
fn test_real_world_preprocessing_sequence() {
    // Full preprocessing sequence as would be used in production

    let n_mels = 80;
    let n_frames = 800; // ~5 seconds at 16kHz with 256 hop

    // 1. Create raw mel spectrogram
    let mut mel = create_realistic_mel(n_mels, n_frames);

    // 2. Z-score normalization
    SciRS2MelOps::normalize_z_score_simd(&mut mel).unwrap();

    // 3. Convert to ndarray for advanced processing
    let mut arr = SciRS2MelOps::to_ndarray(&mel).unwrap();

    // 4. Apply time-domain smoothing (moving average)
    // This would typically be done with ndarray operations
    assert_eq!(arr.shape(), &[n_mels, n_frames]);

    // 5. Convert back to MelSpectrogram
    let processed_mel = SciRS2MelOps::from_ndarray(&arr, 16000);

    // 6. Verify dimensions preserved
    assert_eq!(processed_mel.n_mels, n_mels);
    assert_eq!(processed_mel.n_frames, n_frames);

    // 7. Verify data is reasonable
    let all_values: Vec<f32> = processed_mel.data.iter().flatten().copied().collect();
    assert!(
        all_values.iter().all(|&x| x.is_finite()),
        "All values should be finite"
    );

    let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
    assert!(
        mean.abs() < 0.1,
        "Mean should be close to 0 after normalization"
    );

    println!("✓ Real-world preprocessing sequence test passed");
}

#[test]
fn test_concurrent_model_inference() {
    // Simulate concurrent inference requests from multiple clients

    use std::thread;

    let num_threads = 4;
    let requests_per_thread = 5;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            thread::spawn(move || {
                let mut local_results = Vec::new();

                for req_id in 0..requests_per_thread {
                    // Create mel spectrogram
                    let mut mel = create_realistic_mel(80, 200);

                    // Normalize
                    SciRS2MelOps::normalize_min_max_simd(&mut mel).unwrap();

                    // Store result
                    local_results.push(mel);
                }

                assert_eq!(local_results.len(), requests_per_thread);
                local_results.len()
            })
        })
        .collect();

    let total_processed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    assert_eq!(total_processed, num_threads * requests_per_thread);

    println!("✓ Concurrent model inference test passed");
}

#[test]
fn test_memory_efficient_large_batch() {
    // Test processing very large batch without memory issues

    let batch_size = 100;
    let n_mels = 80;
    let n_frames = 100;

    // Create large batch
    let mut mels: Vec<MelSpectrogram> = (0..batch_size)
        .map(|i| create_realistic_mel(n_mels, n_frames))
        .collect();

    // Process in parallel (should not cause memory issues)
    let result = SciRS2MelOps::batch_normalize_parallel(&mut mels, NormalizationMethod::MinMax);

    assert!(result.is_ok(), "Large batch processing should succeed");
    assert_eq!(mels.len(), batch_size);

    println!("✓ Memory-efficient large batch test passed");
}

#[test]
fn test_deterministic_reproducibility() {
    // Ensure operations are deterministic (important for debugging and testing)

    let n_mels = 80;
    let n_frames = 200;

    // Create identical inputs
    fastrand::seed(12345);
    let mut mel1 = create_realistic_mel(n_mels, n_frames);

    fastrand::seed(12345);
    let mut mel2 = create_realistic_mel(n_mels, n_frames);

    // Apply same operations
    SciRS2MelOps::normalize_z_score_simd(&mut mel1).unwrap();
    SciRS2MelOps::normalize_z_score_simd(&mut mel2).unwrap();

    // Results should be identical
    for (ch1, ch2) in mel1.data.iter().zip(mel2.data.iter()) {
        for (&v1, &v2) in ch1.iter().zip(ch2.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "Results should be deterministic");
        }
    }

    println!("✓ Deterministic reproducibility test passed");
}
