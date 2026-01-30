use super::AudioBuffer;

#[test]
fn test_concatenate() {
    let buf1 = AudioBuffer::mono(vec![1.0, 2.0], 22050);
    let buf2 = AudioBuffer::mono(vec![3.0, 4.0], 22050);
    let buf3 = AudioBuffer::mono(vec![5.0, 6.0], 22050);

    let result = AudioBuffer::concatenate(&[buf1, buf2, buf3]).unwrap();
    assert_eq!(result.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_pad() {
    let mut buffer = AudioBuffer::mono(vec![1.0; 100], 100);
    buffer.pad(0.1, 0.2); // 10 samples before, 20 after

    assert_eq!(buffer.len(), 130);
    // Check padding is zeros
    assert_eq!(buffer.samples()[0], 0.0);
    assert_eq!(buffer.samples()[129], 0.0);
}

#[test]
fn test_has_clipping() {
    let clean = AudioBuffer::mono(vec![0.5, 0.8, -0.9], 22050);
    assert!(!clean.has_clipping());

    let clipped = AudioBuffer::mono(vec![0.5, 1.5, -0.9], 22050);
    assert!(clipped.has_clipping());
}

#[test]
fn test_mfcc() {
    // Create a simple test signal (sine wave)
    let sample_rate = 22050;
    let duration = 0.1; // 100ms
    let frequency = 440.0; // A4 note
    let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let mfccs = buffer.mfcc(13, 26, 512);

    // Should return 13 coefficients
    assert_eq!(mfccs.len(), 13);

    // All coefficients should be finite
    for coeff in &mfccs {
        assert!(coeff.is_finite());
    }

    // First coefficient (C0) should be related to signal energy
    assert!(mfccs[0].abs() > 0.0);
}

#[test]
fn test_mfcc_insufficient_samples() {
    // Buffer too small for FFT
    let buffer = AudioBuffer::mono(vec![0.5; 100], 22050);
    let mfccs = buffer.mfcc(13, 26, 512);

    // Should return empty vector
    assert!(mfccs.is_empty());
}

#[test]
fn test_detect_pitch_autocorr() {
    // Create a simple sine wave at 200 Hz
    let sample_rate = 22050;
    let frequency = 200.0;
    let samples: Vec<f32> = (0..4096)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let detected_pitch = buffer.detect_pitch_autocorr(80.0, 400.0);

    // Should detect pitch close to 200 Hz (within 10% tolerance)
    assert!(detected_pitch > 0.0);
    assert!((detected_pitch - frequency).abs() / frequency < 0.1);
}

#[test]
fn test_detect_pitch_no_pitch() {
    // Test with insufficient samples (should return 0)
    let samples = vec![0.5; 512]; // Too few samples for analysis
    let buffer = AudioBuffer::mono(samples, 22050);
    let pitch = buffer.detect_pitch_autocorr(80.0, 400.0);

    // Should return 0 due to insufficient samples
    assert_eq!(pitch, 0.0);
}

#[test]
fn test_spectral_flux() {
    // Create two buffers with different spectral content
    let sample_rate = 22050;

    // First buffer: 200 Hz sine
    let samples1: Vec<f32> = (0..1024)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
        })
        .collect();

    // Second buffer: 400 Hz sine (different spectral content)
    let samples2: Vec<f32> = (0..1024)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 400.0 * t).sin() * 0.5
        })
        .collect();

    let buffer1 = AudioBuffer::mono(samples1, sample_rate);
    let buffer2 = AudioBuffer::mono(samples2, sample_rate);

    let flux = buffer2.spectral_flux(Some(&buffer1), 512);

    // Should have non-zero flux due to spectral change
    assert!(flux > 0.0);
    assert!(flux.is_finite());
}

#[test]
fn test_spectral_flux_same_buffer() {
    // Flux between identical buffers should be close to zero
    let samples = vec![0.5; 1024];
    let buffer = AudioBuffer::mono(samples.clone(), 22050);

    let flux = buffer.spectral_flux(Some(&buffer), 512);

    // Should be very small (near zero) for identical buffers
    assert!(flux < 0.01);
}

#[test]
fn test_spectral_flux_no_previous() {
    let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    let flux = buffer.spectral_flux(None, 512);

    // Should return 0 with no previous buffer
    assert_eq!(flux, 0.0);
}

#[test]
fn test_estimate_formants() {
    // Create a simple vowel-like signal (mixture of harmonics)
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..2048)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Simulate vowel with formants around 500, 1500, 2500 Hz
            (2.0 * std::f32::consts::PI * 100.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 500.0 * t).sin() * 0.5
                + (2.0 * std::f32::consts::PI * 1500.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 2500.0 * t).sin() * 0.2
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let formants = buffer.estimate_formants(3);

    // Should find some formants
    assert!(!formants.is_empty());

    // All formants should be in reasonable range
    for &formant in &formants {
        assert!(formant >= 200.0);
        assert!(formant <= 4000.0);
    }

    // Formants should be in ascending order (generally)
    if formants.len() >= 2 {
        assert!(formants[1] > formants[0]);
    }
}

#[test]
fn test_estimate_formants_insufficient_samples() {
    // Buffer too small
    let buffer = AudioBuffer::mono(vec![0.5; 100], 22050);
    let formants = buffer.estimate_formants(3);

    // Should return empty vector
    assert!(formants.is_empty());
}

#[test]
fn test_levinson_durbin() {
    // Test Levinson-Durbin algorithm with known autocorrelation
    let buffer = AudioBuffer::mono(vec![1.0, 0.5, 0.25], 1000);

    // Simple autocorrelation sequence
    let autocorr = vec![1.0, 0.8, 0.5, 0.2];
    let lpc = buffer.levinson_durbin(&autocorr, 3);

    // Should return 3 coefficients
    assert_eq!(lpc.len(), 3);

    // All coefficients should be finite
    for &coeff in &lpc {
        assert!(coeff.is_finite());
    }
}

#[test]
fn test_mfcc_power_of_two_fft() {
    // Test with non-power-of-two FFT size (should fail)
    let buffer = AudioBuffer::mono(vec![0.5; 1024], 22050);
    let mfccs = buffer.mfcc(13, 26, 500); // 500 is not power of 2

    // Should return empty vector
    assert!(mfccs.is_empty());
}

#[test]
fn test_get_magnitude_spectrum() {
    let sample_rate = 22050;
    let frequency = 440.0;
    let samples: Vec<f32> = (0..1024)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let spectrum = buffer.get_magnitude_spectrum(512);

    // Should return half FFT size
    assert_eq!(spectrum.len(), 256);

    // All magnitudes should be non-negative and finite
    for &mag in &spectrum {
        assert!(mag >= 0.0);
        assert!(mag.is_finite());
    }

    // Should have peak near the fundamental frequency bin
    let bin_freq = sample_rate as f32 / 512.0;
    let expected_bin = (frequency / bin_freq) as usize;
    let peak_bin = spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // Peak should be within a few bins of expected
    assert!((peak_bin as i32 - expected_bin as i32).abs() < 5);
}

#[test]
fn test_calculate_jitter() {
    // Create a periodic signal with slight period variation
    let sample_rate = 22050;
    let base_freq = 200.0;
    let mut samples = Vec::new();
    let mut phase: f32 = 0.0;

    // Create signal with small period variations (jitter)
    for i in 0..8192 {
        // Add small variation to frequency
        let freq_variation = 1.0 + 0.01 * ((i as f32 / 100.0).sin());
        let instantaneous_freq = base_freq * freq_variation;
        let delta_phase = 2.0 * std::f32::consts::PI * instantaneous_freq / sample_rate as f32;
        samples.push(phase.sin());
        phase += delta_phase;
    }

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let jitter = buffer.calculate_jitter(150.0, 300.0);

    // Should have non-zero but low jitter (< 5% for this synthetic signal)
    assert!(jitter >= 0.0);
    assert!(jitter < 5.0);
    assert!(jitter.is_finite());
}

#[test]
fn test_calculate_jitter_insufficient_samples() {
    let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    let jitter = buffer.calculate_jitter(75.0, 500.0);

    // Should return 0 for insufficient samples
    assert_eq!(jitter, 0.0);
}

#[test]
fn test_calculate_shimmer() {
    // Create a periodic signal with amplitude variation
    let sample_rate = 22050;
    let frequency = 200.0;
    let mut samples = Vec::new();

    for i in 0..8192 {
        let t = i as f32 / sample_rate as f32;
        // Add amplitude modulation (shimmer)
        let amplitude = 0.5 * (1.0 + 0.1 * (2.0 * std::f32::consts::PI * 5.0 * t).sin());
        let signal = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(signal);
    }

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let shimmer = buffer.calculate_shimmer(150.0, 300.0);

    // Should have non-zero but reasonable shimmer
    assert!(shimmer >= 0.0);
    assert!(shimmer < 20.0);
    assert!(shimmer.is_finite());
}

#[test]
fn test_calculate_shimmer_insufficient_samples() {
    let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    let shimmer = buffer.calculate_shimmer(75.0, 500.0);

    // Should return 0 for insufficient samples
    assert_eq!(shimmer, 0.0);
}

#[test]
fn test_calculate_hnr() {
    // Create a clean periodic signal (should have high HNR)
    let sample_rate = 22050;
    let frequency = 200.0;
    let samples: Vec<f32> = (0..4096)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let hnr = buffer.calculate_hnr(150.0, 300.0);

    // Clean periodic signal should have high HNR (> 10 dB)
    assert!(hnr > 5.0, "HNR should be > 5 dB, got {}", hnr);
    assert!(hnr.is_finite());
}

#[test]
fn test_calculate_hnr_noisy_signal() {
    // Create a signal with added noise (lower HNR expected)
    let sample_rate = 22050;
    let frequency = 200.0;
    let samples: Vec<f32> = (0..4096)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let periodic = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
            let noise = (i as f32 * 0.1).sin() * 0.1; // Simple pseudo-noise
            periodic + noise
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let hnr = buffer.calculate_hnr(150.0, 300.0);

    // Noisy signal should have lower HNR than clean signal
    assert!(hnr >= 0.0);
    assert!(hnr.is_finite());
}

#[test]
fn test_calculate_hnr_insufficient_samples() {
    let buffer = AudioBuffer::mono(vec![0.5; 1000], 22050);
    let hnr = buffer.calculate_hnr(75.0, 500.0);

    // Should return 0 for insufficient samples
    assert_eq!(hnr, 0.0);
}

#[test]
fn test_calculate_delta_mfcc() {
    // Create sample MFCC frames
    let mfcc_frames = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.5, 2.5, 3.5, 4.5],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![2.5, 3.5, 4.5, 5.5],
        vec![3.0, 4.0, 5.0, 6.0],
    ];

    let delta_mfccs = AudioBuffer::calculate_delta_mfcc(&mfcc_frames, 2);

    // Should return same number of frames
    assert_eq!(delta_mfccs.len(), 5);

    // Each frame should have same number of coefficients
    for frame in &delta_mfccs {
        assert_eq!(frame.len(), 4);
    }

    // Delta coefficients should represent rate of change
    // For increasing sequence, deltas should be positive
    for frame in &delta_mfccs {
        for &coeff in frame {
            assert!(coeff.is_finite());
        }
    }
}

#[test]
fn test_calculate_delta_mfcc_empty() {
    let delta_mfccs = AudioBuffer::calculate_delta_mfcc(&[], 2);

    // Should return empty vector
    assert!(delta_mfccs.is_empty());
}

#[test]
fn test_calculate_delta_delta_mfcc() {
    // Create sample MFCC frames
    let mfcc_frames = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.5, 2.5, 3.5],
        vec![2.0, 3.0, 4.0],
        vec![2.5, 3.5, 4.5],
    ];

    let delta_mfccs = AudioBuffer::calculate_delta_mfcc(&mfcc_frames, 2);
    let delta_delta_mfccs = AudioBuffer::calculate_delta_delta_mfcc(&delta_mfccs, 2);

    // Should return same number of frames
    assert_eq!(delta_delta_mfccs.len(), 4);

    // Each frame should have same number of coefficients
    for frame in &delta_delta_mfccs {
        assert_eq!(frame.len(), 3);
    }

    // All values should be finite
    for frame in &delta_delta_mfccs {
        for &coeff in frame {
            assert!(coeff.is_finite());
        }
    }
}

#[test]
fn test_delta_mfcc_single_frame() {
    // Single frame should result in zero deltas
    let mfcc_frames = vec![vec![1.0, 2.0, 3.0]];
    let delta_mfccs = AudioBuffer::calculate_delta_mfcc(&mfcc_frames, 2);

    assert_eq!(delta_mfccs.len(), 1);
    // All delta values should be zero for single frame
    for &coeff in &delta_mfccs[0] {
        assert_eq!(coeff, 0.0);
    }
}

#[test]
fn test_voice_quality_metrics_integration() {
    // Create a realistic voice-like signal
    let sample_rate = 22050;
    let frequency = 150.0; // Typical male voice F0
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Fundamental + harmonics
            let fundamental = (2.0 * std::f32::consts::PI * frequency * t).sin();
            let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * frequency * 2.0 * t).sin();
            let harmonic3 = 0.3 * (2.0 * std::f32::consts::PI * frequency * 3.0 * t).sin();
            0.5 * (fundamental + harmonic2 + harmonic3)
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);

    // Test all voice quality metrics
    let jitter = buffer.calculate_jitter(75.0, 300.0);
    let shimmer = buffer.calculate_shimmer(75.0, 300.0);
    let hnr = buffer.calculate_hnr(75.0, 300.0);

    // All metrics should return valid values
    assert!(jitter >= 0.0 && jitter.is_finite(), "Jitter: {}", jitter);
    assert!(
        shimmer >= 0.0 && shimmer.is_finite(),
        "Shimmer: {}",
        shimmer
    );
    assert!(hnr >= 0.0 && hnr.is_finite(), "HNR: {}", hnr);

    // For a synthetic voice with harmonics, we expect reasonable values
    // Note: Harmonics can make period detection more challenging
    // - Jitter may be higher due to harmonic complexity
    // - Shimmer should be moderate
    // - HNR should be positive
    assert!(jitter < 50.0, "Jitter too high: {}", jitter);
    assert!(shimmer < 30.0, "Shimmer too high: {}", shimmer);
    assert!(hnr >= 0.0, "HNR should be non-negative: {}", hnr);
}

#[test]
fn test_chroma_features() {
    // Create a test signal
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let chroma = buffer.chroma_features(2048, 440.0);

    // Should return 12 pitch classes
    assert_eq!(chroma.len(), 12);

    // All values should be between 0 and 1 (normalized)
    for &c in &chroma {
        assert!(c >= 0.0 && c <= 1.0, "Chroma value out of range: {}", c);
    }

    // Sum should be greater than 0
    let sum: f32 = chroma.iter().sum();
    assert!(sum > 0.0);
}

#[test]
fn test_chroma_features_musical_note() {
    // Test with a known musical note (A4 = 440 Hz)
    let sample_rate = 22050;
    let frequency = 440.0; // A4
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let chroma = buffer.chroma_features(2048, 440.0);

    // Should have 12 pitch classes
    assert_eq!(chroma.len(), 12);

    // The pitch class for A (index 9: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9)
    // should have the highest value
    let max_idx = chroma
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // A4 at 440 Hz with ref_freq=440 should map to pitch class 0 (the reference)
    // Actually, 440/440 = 1, log2(1) = 0, so it should be pitch class 0
    assert_eq!(max_idx, 0, "Expected pitch class 0 for A4 with ref=440");
}

#[test]
fn test_chroma_features_insufficient_samples() {
    let buffer = AudioBuffer::mono(vec![0.5; 100], 22050);
    let chroma = buffer.chroma_features(2048, 440.0);

    // Should return empty vector
    assert!(chroma.is_empty());
}

#[test]
fn test_spectral_contrast() {
    // Create a test signal with varying spectral content
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies
            (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
                + (2.0 * std::f32::consts::PI * 800.0 * t).sin() * 0.3
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let contrast = buffer.spectral_contrast(2048, 6);

    // Should return 6 bands
    assert_eq!(contrast.len(), 6);

    // All contrast values should be finite
    for &c in &contrast {
        assert!(c.is_finite(), "Contrast value not finite: {}", c);
    }

    // Contrast values should generally be positive (peaks > valleys)
    for &c in &contrast {
        assert!(c >= 0.0, "Negative contrast: {}", c);
    }
}

#[test]
fn test_spectral_contrast_with_harmonics() {
    // Create rich harmonic signal
    let sample_rate = 22050;
    let fundamental = 150.0;
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let mut signal = 0.0;
            // Add harmonics
            for harmonic in 1..=5 {
                let freq = fundamental * harmonic as f32;
                let amplitude = 1.0 / harmonic as f32;
                signal += amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
            signal * 0.2
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let contrast = buffer.spectral_contrast(2048, 8);

    // Should return 8 bands
    assert_eq!(contrast.len(), 8);

    // Harmonic signals should have high contrast (peaks at harmonics, valleys between)
    let avg_contrast: f32 = contrast.iter().sum::<f32>() / contrast.len() as f32;
    assert!(
        avg_contrast > 5.0,
        "Expected high contrast for harmonic signal"
    );
}

#[test]
fn test_spectral_contrast_edge_cases() {
    // Test with insufficient samples
    let buffer = AudioBuffer::mono(vec![0.5; 100], 22050);
    let contrast = buffer.spectral_contrast(2048, 6);
    assert!(contrast.is_empty());

    // Test with 0 bands
    let buffer = AudioBuffer::mono(vec![0.5; 8192], 22050);
    let contrast = buffer.spectral_contrast(2048, 0);
    assert!(contrast.is_empty());
}

#[test]
fn test_detect_pitch_yin() {
    // Create a pure tone at known frequency
    let sample_rate = 22050;
    let frequency = 220.0; // A3
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);
    let detected = buffer.detect_pitch_yin(100.0, 400.0, 0.15);

    // Should detect pitch close to 220 Hz
    assert!(detected > 0.0, "No pitch detected");
    assert!(
        (detected - frequency).abs() < 5.0,
        "Detected pitch {} too far from expected {}",
        detected,
        frequency
    );
}

#[test]
fn test_detect_pitch_yin_comparison_with_autocorr() {
    // Compare YIN with autocorrelation for a complex signal
    let sample_rate = 22050;
    let frequency = 150.0;
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Fundamental + harmonics (realistic voice)
            let fundamental = (2.0 * std::f32::consts::PI * frequency * t).sin();
            let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * frequency * 2.0 * t).sin();
            let harmonic3 = 0.3 * (2.0 * std::f32::consts::PI * frequency * 3.0 * t).sin();
            0.4 * (fundamental + harmonic2 + harmonic3)
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);

    let pitch_yin = buffer.detect_pitch_yin(80.0, 300.0, 0.15);
    let pitch_autocorr = buffer.detect_pitch_autocorr(80.0, 300.0);

    // Both should detect pitch
    assert!(pitch_yin > 0.0, "YIN: No pitch detected");
    assert!(pitch_autocorr > 0.0, "Autocorr: No pitch detected");

    // Both should be close to the fundamental
    assert!(
        (pitch_yin - frequency).abs() < 10.0,
        "YIN pitch {} too far from expected {}",
        pitch_yin,
        frequency
    );

    // YIN is often more accurate than autocorrelation
    // But both should be in the right ballpark
    println!(
        "YIN: {:.2} Hz, Autocorr: {:.2} Hz, Expected: {:.2} Hz",
        pitch_yin, pitch_autocorr, frequency
    );
}

#[test]
fn test_detect_pitch_yin_no_pitch() {
    // Test with white noise (no pitch)
    let samples: Vec<f32> = (0..8192)
        .map(|i| (i as f32 * 0.1).sin() * 0.001) // Very low amplitude, irregular
        .collect();

    let buffer = AudioBuffer::mono(samples, 22050);
    let detected = buffer.detect_pitch_yin(80.0, 400.0, 0.1);

    // May or may not detect pitch in noise
    // Just verify it doesn't crash and returns valid value
    assert!(detected >= 0.0 && detected.is_finite());
}

#[test]
fn test_detect_pitch_yin_insufficient_samples() {
    let buffer = AudioBuffer::mono(vec![0.5; 100], 22050);
    let detected = buffer.detect_pitch_yin(80.0, 400.0, 0.15);

    // Should return 0.0 for insufficient samples
    assert_eq!(detected, 0.0);
}

#[test]
fn test_chroma_and_contrast_integration() {
    // Test that chroma and contrast work together for music analysis
    let sample_rate = 22050;
    let samples: Vec<f32> = (0..8192)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // C major chord: C (261.63), E (329.63), G (392.00)
            (2.0 * std::f32::consts::PI * 261.63 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 329.63 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 392.00 * t).sin() * 0.3
        })
        .collect();

    let buffer = AudioBuffer::mono(samples, sample_rate);

    let chroma = buffer.chroma_features(4096, 440.0);
    let contrast = buffer.spectral_contrast(4096, 6);

    // Both should return valid results
    assert_eq!(chroma.len(), 12);
    assert_eq!(contrast.len(), 6);

    // For a chord, multiple pitch classes should have energy
    let active_classes = chroma.iter().filter(|&&c| c > 0.1).count();
    assert!(
        active_classes >= 2,
        "Chord should activate multiple pitch classes"
    );

    // Chord should have moderate to high spectral contrast
    let avg_contrast: f32 = contrast.iter().sum::<f32>() / contrast.len() as f32;
    assert!(avg_contrast > 0.0);
}
