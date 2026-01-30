//! Audio analysis utility functions for real-time emotion adaptation
//!
//! This module provides helper functions for analyzing audio characteristics
//! that are used to adapt emotional expression in real-time synthesis.

/// Calculate RMS (Root Mean Square) energy of audio samples
///
/// # Arguments
/// * `samples` - Audio sample buffer
///
/// # Returns
/// RMS energy value (0.0 to 1.0+)
pub(crate) fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Calculate zero crossing rate of audio samples
///
/// Zero crossing rate is a measure of noisiness/high-frequency content.
///
/// # Arguments
/// * `samples` - Audio sample buffer
///
/// # Returns
/// Zero crossing rate (0.0 to 1.0)
pub(crate) fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let crossings = samples.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

    crossings as f32 / samples.len() as f32
}

/// Calculate spectral centroid of audio samples
///
/// Spectral centroid is a measure of brightness/timbral quality.
///
/// # Arguments
/// * `samples` - Audio sample buffer
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Normalized spectral centroid (0.0 to 1.0)
pub(crate) fn calculate_spectral_centroid(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.len() < 64 {
        return 0.5; // Default value for very short signals
    }

    // Use a simple sliding window approach to estimate spectral centroid
    let window_size = 32;
    let mut centroids = Vec::new();

    for start in (0..samples.len()).step_by(window_size / 2) {
        let end = (start + window_size).min(samples.len());
        let window = &samples[start..end];

        if window.len() >= 4 {
            let centroid = calculate_window_spectral_centroid(window);
            centroids.push(centroid);
        }
    }

    if centroids.is_empty() {
        return 0.5;
    }

    // Average the centroids and normalize
    let avg_centroid = centroids.iter().sum::<f32>() / centroids.len() as f32;

    // Convert to normalized frequency (0.0 to 1.0)
    (avg_centroid * sample_rate * 0.5).clamp(0.0, 1.0)
}

/// Calculate spectral centroid for a small window using frequency domain approximation
///
/// # Arguments
/// * `window` - Audio window buffer
///
/// # Returns
/// Window spectral centroid
pub(crate) fn calculate_window_spectral_centroid(window: &[f32]) -> f32 {
    // Calculate spectral centroid for a small window using frequency domain approximation
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;

    // Simple frequency domain analysis using differences
    for i in 1..window.len() {
        let freq_weight = i as f32 / window.len() as f32;
        let magnitude = (window[i] - window[i - 1]).abs();

        weighted_sum += freq_weight * magnitude;
        magnitude_sum += magnitude;
    }

    if magnitude_sum > 0.0001 {
        weighted_sum / magnitude_sum
    } else {
        0.5
    }
}

/// Calculate spectral rolloff of audio samples
///
/// Spectral rolloff is the frequency below which 85% of the signal energy is contained.
///
/// # Arguments
/// * `samples` - Audio sample buffer
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Normalized spectral rolloff frequency (0.0 to 1.0)
pub(crate) fn calculate_spectral_rolloff(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.len() < 16 {
        return 0.5; // Default value for very short signals
    }

    // Calculate energy distribution across frequency bins using a simple approximation
    let bin_count = 16; // Use 16 frequency bins
    let mut energy_bins = vec![0.0; bin_count];
    let window_size = samples.len() / bin_count;

    // Distribute energy across bins based on signal characteristics
    for (bin_idx, bin_energy) in energy_bins.iter_mut().enumerate() {
        let start = bin_idx * window_size;
        let end = ((bin_idx + 1) * window_size).min(samples.len());

        if start < end {
            let window = &samples[start..end];

            // Calculate high-frequency content using derivatives
            let mut high_freq_energy = 0.0;
            for i in 1..window.len() {
                let diff = (window[i] - window[i - 1]).abs();
                high_freq_energy += diff * diff;
            }

            // Weight by frequency bin
            let freq_weight = (bin_idx as f32 + 1.0) / bin_count as f32;
            *bin_energy = high_freq_energy * freq_weight;
        }
    }

    // Find 85th percentile energy threshold
    let total_energy: f32 = energy_bins.iter().sum();
    if total_energy < 0.0001 {
        return 0.5; // No significant energy
    }

    let target_energy = total_energy * 0.85;
    let mut cumulative_energy = 0.0;

    for (bin_idx, &bin_energy) in energy_bins.iter().enumerate() {
        cumulative_energy += bin_energy;
        if cumulative_energy >= target_energy {
            // Return normalized frequency (0.0 to 1.0)
            return (bin_idx as f32 / bin_count as f32).min(1.0);
        }
    }

    // If we reach here, most energy is in high frequencies
    0.9
}

/// Calculate tempo strength using onset detection
///
/// Returns a value between 0.0 (no rhythmic structure) and 1.0 (strong rhythm)
///
/// # Arguments
/// * `samples` - Audio sample buffer
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Tempo strength (0.0 to 1.0)
pub(crate) fn calculate_tempo_strength(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.len() < 512 {
        return 0.0; // Need minimum samples for tempo analysis
    }

    // Calculate energy envelope using frame-based RMS
    let frame_size = (sample_rate * 0.023) as usize; // ~23ms frames (typical for onset detection)
    let hop_size = frame_size / 2; // 50% overlap

    let mut energy_envelope = Vec::new();
    let mut pos = 0;

    while pos + frame_size <= samples.len() {
        let frame = &samples[pos..pos + frame_size];
        let frame_energy = calculate_rms(frame);
        energy_envelope.push(frame_energy);
        pos += hop_size;
    }

    if energy_envelope.len() < 4 {
        return 0.0;
    }

    // Detect onsets using first-order difference (spectral flux approximation)
    let mut onset_strength = Vec::new();
    for i in 1..energy_envelope.len() {
        let diff = (energy_envelope[i] - energy_envelope[i - 1]).max(0.0); // Positive differences only
        onset_strength.push(diff);
    }

    // Calculate regularity using autocorrelation
    let lag_range = onset_strength.len() / 2;
    let mut max_correlation = 0.0;

    // Normalize onset strength
    let onset_mean: f32 = onset_strength.iter().sum::<f32>() / onset_strength.len() as f32;
    let onset_std = (onset_strength
        .iter()
        .map(|x| (x - onset_mean).powi(2))
        .sum::<f32>()
        / onset_strength.len() as f32)
        .sqrt();

    if onset_std < 0.0001 {
        return 0.0; // No variation in onsets
    }

    // Compute autocorrelation for different lags
    for lag in 2..lag_range.min(100) {
        // Focus on musically relevant tempos
        let mut correlation = 0.0;
        let mut count = 0;

        for i in 0..(onset_strength.len() - lag) {
            correlation +=
                (onset_strength[i] - onset_mean) * (onset_strength[i + lag] - onset_mean);
            count += 1;
        }

        if count > 0 {
            correlation /= count as f32 * onset_std * onset_std;

            if correlation > max_correlation {
                max_correlation = correlation;
            }
        }
    }

    // Tempo strength is based on:
    // 1. Maximum autocorrelation value (periodicity strength)
    // 2. Variance in onset strengths (rhythm clarity)
    let periodicity_score = max_correlation.clamp(0.0, 1.0);
    let clarity_score = (onset_std / (onset_mean + 0.001)).min(1.0);

    // Combine scores
    (periodicity_score * 0.7 + clarity_score * 0.3).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_rms() {
        // Test empty samples
        assert_eq!(calculate_rms(&[]), 0.0);

        // Test constant value
        let samples = vec![0.5; 100];
        let rms = calculate_rms(&samples);
        assert!((rms - 0.5).abs() < 0.001);

        // Test alternating values
        let samples: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let rms = calculate_rms(&samples);
        assert!((rms - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_zero_crossing_rate() {
        // Test empty samples
        assert_eq!(calculate_zero_crossing_rate(&[]), 0.0);

        // Test single sample
        assert_eq!(calculate_zero_crossing_rate(&[0.5]), 0.0);

        // Test no crossings
        let samples = vec![0.5; 100];
        assert_eq!(calculate_zero_crossing_rate(&samples), 0.0);

        // Test maximum crossings (alternating)
        let samples: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let zcr = calculate_zero_crossing_rate(&samples);
        assert!(zcr > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_calculate_spectral_centroid() {
        // Test short signal (should return default)
        let short_samples = vec![0.5; 32];
        let centroid = calculate_spectral_centroid(&short_samples, 44100.0);
        assert_eq!(centroid, 0.5);

        // Test longer signal
        let samples = vec![0.5; 128];
        let centroid = calculate_spectral_centroid(&samples, 44100.0);
        assert!(centroid >= 0.0 && centroid <= 1.0);
    }

    #[test]
    fn test_calculate_spectral_rolloff() {
        // Test short signal (should return default)
        let short_samples = vec![0.5; 8];
        let rolloff = calculate_spectral_rolloff(&short_samples, 44100.0);
        assert_eq!(rolloff, 0.5);

        // Test longer signal
        let samples = vec![0.5; 256];
        let rolloff = calculate_spectral_rolloff(&samples, 44100.0);
        assert!(rolloff >= 0.0 && rolloff <= 1.0);
    }

    #[test]
    fn test_calculate_tempo_strength() {
        // Test short signal (should return 0.0)
        let short_samples = vec![0.5; 256];
        let tempo = calculate_tempo_strength(&short_samples, 44100.0);
        assert_eq!(tempo, 0.0);

        // Test longer signal
        let samples = vec![0.5; 2048];
        let tempo = calculate_tempo_strength(&samples, 44100.0);
        assert!(tempo >= 0.0 && tempo <= 1.0);
    }

    #[test]
    fn test_window_spectral_centroid() {
        let window = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let centroid = calculate_window_spectral_centroid(&window);
        assert!(centroid >= 0.0 && centroid <= 1.0);

        // Test silence
        let silence = vec![0.0; 10];
        let centroid = calculate_window_spectral_centroid(&silence);
        assert_eq!(centroid, 0.5);
    }
}
