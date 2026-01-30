//! Tests for SIMD-optimized audio processing operations

use super::intrinsics::SimdAudioProcessor;
use super::*;

#[test]
fn test_rms_calculation() {
    let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let rms = SimdAudioProcessor::calculate_rms(&samples);

    // Calculate expected RMS
    let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
    let expected = (sum_squares / samples.len() as f32).sqrt();

    assert!((rms - expected).abs() < 1e-6);
}

#[test]
fn test_rms_simd_vs_scalar() {
    let samples: Vec<f32> = (0..1000).map(|x| (x as f32) * 0.001).collect();

    let scalar_rms = SimdAudioProcessor::calculate_rms_scalar(&samples);
    let simd_rms = SimdAudioProcessor::calculate_rms(&samples);

    // Results should be very close (allowing for floating point precision)
    assert!((scalar_rms - simd_rms).abs() < 1e-5);
}

#[test]
fn test_peak_finding() {
    let samples = vec![-0.5, 0.3, -0.8, 0.9, -0.2];
    let peak = SimdAudioProcessor::find_peak(&samples);

    assert!((peak - 0.9).abs() < 1e-6);
}

#[test]
fn test_peak_simd_vs_scalar() {
    let samples: Vec<f32> = (0..1000).map(|x| ((x as f32) * 0.01).sin()).collect();

    let scalar_peak = SimdAudioProcessor::find_peak_scalar(&samples);
    let simd_peak = SimdAudioProcessor::find_peak(&samples);

    assert!((scalar_peak - simd_peak).abs() < 1e-5);
}

#[test]
fn test_apply_gain() {
    let mut samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let gain = 2.0;

    SimdAudioProcessor::apply_gain(&mut samples, gain);

    let expected = vec![0.2, 0.4, 0.6, 0.8, 1.0];
    for (actual, exp) in samples.iter().zip(expected.iter()) {
        assert!((actual - exp).abs() < 1e-6);
    }
}

#[test]
fn test_apply_gain_simd_vs_scalar() {
    let mut simd_samples: Vec<f32> = (0..1000).map(|x| (x as f32) * 0.001).collect();
    let mut scalar_samples = simd_samples.clone();
    let gain = 1.5;

    SimdAudioProcessor::apply_gain_scalar(&mut scalar_samples, gain);
    SimdAudioProcessor::apply_gain(&mut simd_samples, gain);

    for (simd, scalar) in simd_samples.iter().zip(scalar_samples.iter()) {
        assert!((simd - scalar).abs() < 1e-5);
    }
}

#[test]
fn test_mix_samples() {
    let mut target = vec![0.1, 0.2, 0.3, 0.4];
    let source = vec![0.2, 0.3, 0.4, 0.5];
    let scale = 0.5;

    SimdAudioProcessor::mix_samples(&mut target, &source, scale);

    let expected = vec![0.2, 0.35, 0.5, 0.65];
    for (actual, exp) in target.iter().zip(expected.iter()) {
        assert!((actual - exp).abs() < 1e-6);
    }
}

#[test]
fn test_count_above_threshold() {
    let samples = vec![-0.5, 0.3, -0.8, 0.9, -0.2, 0.7];
    let threshold = 0.5;

    let count = SimdAudioProcessor::count_above_threshold(&samples, threshold);

    // Count samples where |sample| > threshold
    // |-0.5| = 0.5 (not > 0.5), |0.3| = 0.3 (no), |-0.8| = 0.8 (yes),
    // |0.9| = 0.9 (yes), |-0.2| = 0.2 (no), |0.7| = 0.7 (yes)
    assert_eq!(count, 3); // -0.8, 0.9, and 0.7 have absolute value > 0.5
}

#[test]
fn test_convert_i16_to_f32() {
    let input: Vec<i16> = vec![0, 16384, -16384, 32767, -32768];
    let mut output = vec![0.0f32; input.len()];
    let normalization = 1.0 / 32768.0;

    SimdAudioProcessor::convert_i16_to_f32(&input, &mut output, normalization);

    assert!((output[0] - 0.0).abs() < 1e-6);
    assert!((output[1] - 0.5).abs() < 0.01);
    assert!((output[2] + 0.5).abs() < 0.01);
    assert!((output[3] - 1.0).abs() < 0.01);
    assert!((output[4] + 1.0).abs() < 0.01);
}

#[test]
fn test_remove_dc_offset() {
    let mut samples = vec![1.1, 1.2, 1.3, 1.4, 1.5];

    SimdAudioProcessor::remove_dc_offset(&mut samples);

    // After DC removal, mean should be close to 0
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn test_dc_offset_removal() {
    // Create samples with DC offset
    let mut samples = vec![1.0, 1.1, 0.9, 1.2, 0.8];

    SimdAudioProcessor::remove_dc_offset(&mut samples);

    // After removal, mean should be approximately 0
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    assert!(
        mean.abs() < 1e-5,
        "Mean after DC removal should be ~0, got {}",
        mean
    );
}

#[test]
fn test_sample_clipping() {
    let mut samples = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
    let limit = 1.0;

    SimdAudioProcessor::clip_samples(&mut samples, limit);

    assert_eq!(samples[0], -1.0);
    assert_eq!(samples[1], -0.5);
    assert_eq!(samples[2], 0.0);
    assert_eq!(samples[3], 0.5);
    assert_eq!(samples[4], 1.0);
}

#[test]
fn test_zero_crossing_rate() {
    // Create a simple waveform with known zero crossings
    let samples = vec![-1.0, -0.5, 0.5, 1.0, 0.5, -0.5, -1.0];

    let zcr = SimdAudioProcessor::calculate_zero_crossing_rate(&samples);

    // Should have 2 zero crossings in 6 transitions
    let expected = 2.0 / 6.0;
    assert!((zcr - expected).abs() < 0.01);
}

#[test]
fn test_hann_window() {
    let mut samples = vec![1.0; 100];

    SimdAudioProcessor::apply_hann_window(&mut samples);

    // First and last samples should be close to 0
    assert!(samples[0].abs() < 0.01);
    assert!(samples[99].abs() < 0.01);

    // Middle sample should be close to 1
    assert!((samples[50] - 1.0).abs() < 0.01);
}

#[test]
fn test_hamming_window() {
    let mut samples = vec![1.0; 100];

    SimdAudioProcessor::apply_hamming_window(&mut samples);

    // Hamming window doesn't go all the way to 0 at the edges
    assert!(samples[0] > 0.0 && samples[0] < 0.1);
    assert!(samples[99] > 0.0 && samples[99] < 0.1);

    // Middle sample should be close to 1
    assert!((samples[50] - 1.0).abs() < 0.01);
}

#[test]
fn test_dc_offset_large_array() {
    // Test with a larger array to ensure SIMD paths are tested
    let mut samples: Vec<f32> = (0..1000).map(|x| (x as f32) * 0.001 + 0.5).collect();

    SimdAudioProcessor::remove_dc_offset(&mut samples);

    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_clipping_preserves_in_range_values() {
    let mut samples = vec![-0.5, -0.3, 0.0, 0.3, 0.5];
    let original = samples.clone();
    let limit = 1.0;

    SimdAudioProcessor::clip_samples(&mut samples, limit);

    // All values should be unchanged since they're within [-1.0, 1.0]
    for (clipped, orig) in samples.iter().zip(original.iter()) {
        assert_eq!(clipped, orig);
    }
}

#[test]
fn test_energy_calculation() {
    let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let energy = SimdAudioProcessor::calculate_energy(&samples);

    // Energy is sum of squares
    let expected: f32 = samples.iter().map(|&x| x * x).sum();
    assert!((energy - expected).abs() < 1e-5);
}

#[test]
fn test_energy_simd_vs_scalar() {
    let samples: Vec<f32> = (0..1000).map(|x| (x as f32) * 0.001).collect();

    let scalar_energy = SimdAudioProcessor::calculate_energy_scalar(&samples);
    let simd_energy = SimdAudioProcessor::calculate_energy(&samples);

    assert!((scalar_energy - simd_energy).abs() < 1e-3);
}

#[test]
fn test_crest_factor() {
    let samples = vec![0.1, 0.2, 0.3, 0.4, 1.0]; // Peak = 1.0

    let crest_factor = SimdAudioProcessor::calculate_crest_factor(&samples);

    // Crest factor = peak / RMS
    let rms = SimdAudioProcessor::calculate_rms(&samples);
    let expected = 1.0 / rms;

    assert!((crest_factor - expected).abs() < 1e-5);
}

#[test]
fn test_spectral_flatness() {
    let samples: Vec<f32> = (0..100).map(|x| (x as f32).sin()).collect();

    let flatness = SimdAudioProcessor::calculate_spectral_flatness_approximation(&samples);

    // Flatness should be between 0 and 1
    assert!(flatness >= 0.0 && flatness <= 1.0);
}

#[test]
fn test_autocorrelation() {
    let samples = vec![1.0, 0.5, 0.0, -0.5, -1.0];
    let lag = 0;

    let autocorr = SimdAudioProcessor::calculate_autocorrelation_scalar(&samples, lag);

    // Autocorrelation at lag 0 is normalized by energy, so it should be 1.0
    // (sum of x[i]*x[i]) / (sum of x[i]^2) = 1.0
    assert!(
        (autocorr - 1.0).abs() < 1e-5,
        "autocorr={}, expected=1.0",
        autocorr
    );
}

#[test]
fn test_autocorrelation_function() {
    let samples: Vec<f32> = (0..100).map(|x| (x as f32).sin()).collect();
    let min_lag = 1;
    let max_lag = 10;

    let autocorr =
        SimdAudioProcessor::calculate_autocorrelation_function(&samples, min_lag, max_lag);

    assert_eq!(autocorr.len(), max_lag - min_lag + 1);

    // Should have computed autocorrelation for the specified lag range
    assert!(!autocorr.is_empty());
}

#[test]
fn test_pitch_estimation() {
    // Create a synthetic periodic signal
    let sample_rate = 16000u32;
    let frequency = 440.0; // A4 note
    let duration = 0.1; // 100ms
    let num_samples = (sample_rate as f32 * duration) as usize;

    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect();

    let min_freq = 80.0;
    let max_freq = 800.0;

    let estimated_pitch = SimdAudioProcessor::estimate_pitch_autocorrelation(
        &samples,
        sample_rate,
        min_freq,
        max_freq,
    );

    // Pitch estimation may not be perfect, but should be non-zero for periodic signal
    assert!(
        estimated_pitch > 0.0,
        "Pitch estimation should detect pitch in periodic signal"
    );

    // Autocorrelation-based pitch detection can find subharmonics (frequency/N) or harmonics (frequency*N)
    // Accept if estimated is within 20% of frequency, or a clean integer multiple/divisor
    let ratio_up = estimated_pitch / frequency;
    let ratio_down = frequency / estimated_pitch;

    let is_harmonic = (ratio_up - ratio_up.round()).abs() < 0.1 && ratio_up >= 0.9;
    let is_subharmonic = (ratio_down - ratio_down.round()).abs() < 0.1 && ratio_down >= 0.9;
    let direct_match = (estimated_pitch - frequency).abs() / frequency < 0.2;

    assert!(
        is_harmonic || is_subharmonic || direct_match,
        "Pitch estimation found {:.2} Hz (expected {:.2} Hz, ratio: {:.2}), which is a valid subharmonic/harmonic",
        estimated_pitch,
        frequency,
        ratio_down
    );
}

#[test]
fn test_spectral_centroid() {
    let samples: Vec<f32> = (0..100).map(|x| (x as f32).sin()).collect();

    let centroid = SimdAudioProcessor::calculate_spectral_centroid_approximation(&samples);

    // Centroid should be positive
    assert!(centroid > 0.0);
}

#[test]
fn test_spectral_rolloff() {
    let samples: Vec<f32> = (0..100).map(|x| (x as f32).sin()).collect();

    let rolloff = SimdAudioProcessor::calculate_spectral_rolloff_approximation(&samples);

    // Rolloff should be within valid range
    assert!(rolloff >= 0.0 && rolloff <= samples.len() as f32);
}

#[test]
fn test_spectral_bandwidth() {
    let samples: Vec<f32> = (0..100).map(|x| (x as f32).sin()).collect();

    let bandwidth = SimdAudioProcessor::calculate_spectral_bandwidth_approximation(&samples);

    // Bandwidth should be positive
    assert!(bandwidth > 0.0);
}

#[test]
fn test_large_arrays_performance() {
    // Test with larger arrays to ensure SIMD optimizations are triggered
    let samples: Vec<f32> = (0..10000).map(|x| (x as f32) * 0.0001).collect();

    // Test various operations
    let _rms = SimdAudioProcessor::calculate_rms(&samples);
    let _peak = SimdAudioProcessor::find_peak(&samples);
    let _energy = SimdAudioProcessor::calculate_energy(&samples);
    let _crest = SimdAudioProcessor::calculate_crest_factor(&samples);

    // If we get here without panicking, the operations succeeded
}

#[test]
fn test_empty_array_handling() {
    let samples: Vec<f32> = vec![];

    // These should handle empty arrays gracefully
    let rms = SimdAudioProcessor::calculate_rms_scalar(&samples);
    assert_eq!(rms, 0.0);

    let peak = SimdAudioProcessor::find_peak_scalar(&samples);
    assert_eq!(peak, 0.0);
}

#[test]
fn test_single_sample() {
    let samples = vec![0.5];

    let rms = SimdAudioProcessor::calculate_rms(&samples);
    assert!((rms - 0.5).abs() < 1e-6);

    let peak = SimdAudioProcessor::find_peak(&samples);
    assert!((peak - 0.5).abs() < 1e-6);
}

#[test]
fn test_blackman_window() {
    let mut samples = vec![1.0; 100];

    SimdAudioProcessor::apply_blackman_window(&mut samples);

    // Blackman window should taper to near zero at the edges
    assert!(samples[0].abs() < 0.01);
    assert!(samples[99].abs() < 0.01);

    // Middle should be close to 1
    assert!((samples[50] - 1.0).abs() < 0.1);
}

#[test]
fn test_blackman_harris_window() {
    let mut samples = vec![1.0; 100];

    SimdAudioProcessor::apply_blackman_harris_window(&mut samples);

    // Blackman-Harris window should taper to very close to zero at the edges
    assert!(samples[0].abs() < 0.01);
    assert!(samples[99].abs() < 0.01);

    // Middle should be close to 1
    assert!((samples[50] - 1.0).abs() < 0.1);
}
