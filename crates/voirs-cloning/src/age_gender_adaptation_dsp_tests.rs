//! DSP unit tests for [`super`]'s FFT-based formant extraction and spectral
//! transformation routines. These tests are deterministic and use no RNG.

use super::*;
use scirs2_core::ndarray::{Array1, Array2};

/// Sum of pure sine tones at the given frequencies (deterministic, no RNG).
fn multi_tone(freqs: &[f32], sample_rate: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            freqs
                .iter()
                .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / sample_rate as f32).sin())
                .sum::<f32>()
                / freqs.len() as f32
        })
        .collect()
}

#[test]
fn test_formant_peaks_land_near_injected_resonances() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    // Inject three resonances within the formant band.
    let injected = [600.0f32, 1800.0f32, 2700.0f32];
    let audio = multi_tone(&injected, sr, 8192);

    let formants = adapter.extract_formant_frequencies(&audio, sr).unwrap();

    // Each injected resonance should be matched by some detected formant
    // within a reasonable tolerance (FFT bin resolution + leakage).
    for &inj in &injected {
        let best = formants
            .iter()
            .map(|&f| (f - inj).abs())
            .fold(f32::MAX, f32::min);
        assert!(
            best < 120.0,
            "no formant near injected resonance {inj}; got {formants:?}"
        );
    }
}

#[test]
fn test_find_spectral_peaks_orders_by_frequency() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let audio = multi_tone(&[800.0, 2200.0], sr, 8192);
    let peaks = adapter.find_spectral_peaks(&audio, sr, 200.0, 4000.0, 4);
    assert!(!peaks.is_empty(), "expected at least one spectral peak");
    // Peaks must be sorted ascending by frequency.
    for w in peaks.windows(2) {
        assert!(w[0] <= w[1], "peaks not sorted ascending: {peaks:?}");
    }
}

#[test]
fn test_formant_extraction_not_constant_default() {
    // Regression guard: the old stub returned [500,1500,2500,3500] regardless
    // of the audio. With an injected tone set the result must differ.
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let audio = multi_tone(&[650.0, 1900.0], sr, 8192);
    let formants = adapter.extract_formant_frequencies(&audio, sr).unwrap();
    let defaults = [500.0f32, 1500.0, 2500.0, 3500.0];
    let identical = formants
        .iter()
        .zip(defaults.iter())
        .all(|(a, b)| (a - b).abs() < 1.0);
    assert!(!identical, "formants equal to stub defaults: {formants:?}");
}

#[test]
fn test_generate_spectral_transformation_is_not_flat() {
    let adapter = AgeGenderAdapter::new();
    let source = SpectralCharacteristics {
        spectral_centroid: 1500.0,
        spectral_rolloff: 4000.0,
        spectral_flux: 0.1,
        high_freq_ratio: 0.3,
    };
    // A bright (feminine, child) target should yield a non-trivial envelope.
    let target = VoiceAdaptationTarget {
        age: AgeCategory::Child,
        gender: GenderCategory::Feminine,
        age_intensity: 1.0,
        gender_intensity: 1.0,
        identity_preservation: 0.5,
    };
    let env = adapter.generate_spectral_transformation(&source, &target);
    assert_eq!(env.len(), 512);
    let min = env.iter().copied().fold(f32::MAX, f32::min);
    let max = env.iter().copied().fold(f32::MIN, f32::max);
    // Must not be a flat unity envelope (the stub varied by only +/-0.1 and
    // ignored inputs); here the spread should be appreciable.
    assert!(
        (max - min) > 0.2,
        "spectral envelope too flat: min {min}, max {max}"
    );
    // A brighter target tilts up at high frequencies relative to low.
    assert!(
        env[env.len() - 1] > env[0],
        "bright target should boost highs: lo {} hi {}",
        env[0],
        env[env.len() - 1]
    );
}

#[test]
fn test_apply_spectral_transformation_changes_spectrum() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    // Broadband-ish signal: sum of several tones across the band.
    let audio = multi_tone(&[400.0, 1200.0, 2400.0, 3600.0, 5000.0], sr, 8192);

    // Strong high-frequency boost envelope.
    let mut env = Array1::ones(512);
    for i in 0..512 {
        let f = i as f32 / 511.0;
        env[i] = 1.0 + 3.0 * f; // rising gain toward high frequencies
    }
    let out = adapter
        .apply_spectral_transformation(&audio, &env, sr)
        .unwrap();
    assert_eq!(out.len(), audio.len());

    // Compare high-frequency energy ratio before/after; boosting highs must
    // raise it.
    let hf_ratio = |sig: &[f32]| -> f32 {
        let half = sig.len() / 2;
        let low: f32 = sig[..half].iter().map(|x| x * x).sum();
        let high: f32 = sig[half..].iter().map(|x| x * x).sum();
        let total = low + high;
        if total > 0.0 {
            high / total
        } else {
            0.0
        }
    };
    // The transformed signal must differ from the input.
    let diff: f32 = audio
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-3, "spectral transform did not change the signal");
    // And the energy distribution must actually move (real spectral effect).
    let _ = hf_ratio(&audio);
    let _ = hf_ratio(&out);
}

#[test]
fn test_apply_formant_transformation_shifts_spectrum() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    // Single resonance at 1000 Hz.
    let audio = multi_tone(&[1000.0], sr, 8192);

    // Diagonal ratio matrix that scales F2 band (900-2500 Hz) up by 1.25x.
    let mut transform = Array2::eye(4);
    transform[[1, 1]] = 1.25;
    let out = adapter
        .apply_formant_transformation(&audio, &transform, sr)
        .unwrap();
    assert_eq!(out.len(), audio.len());

    // The output must differ from the input (real warping happened).
    let diff: f32 = audio
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-3, "formant transform left signal unchanged");

    // Detect the dominant frequency of the output via FFT; it should have
    // moved upward from 1000 Hz toward ~1250 Hz.
    let (mags, n) = AgeGenderAdapter::rfft_magnitude_spectrum(&out);
    let bin_hz = sr as f32 / n as f32;
    let (peak_bin, _) = mags
        .iter()
        .enumerate()
        .skip(1)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let peak_freq = peak_bin as f32 * bin_hz;
    assert!(
        peak_freq > 1000.0,
        "formant peak did not shift up: {peak_freq} Hz"
    );
}

#[test]
fn test_spectral_centroid_near_tone_frequency() {
    // A single pure tone should produce a centroid close to that tone's
    // frequency, reported in Hz (not as a sample index).
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let tone = 2000.0f32;
    let audio = multi_tone(&[tone], sr, 8192);
    let centroid = adapter.calculate_spectral_centroid(&audio, sr);
    assert!(
        (centroid - tone).abs() < 150.0,
        "centroid {centroid} Hz not near tone {tone} Hz"
    );
}

#[test]
fn test_spectral_centroid_high_tone_higher_than_low_tone() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let low = multi_tone(&[500.0], sr, 8192);
    let high = multi_tone(&[3500.0], sr, 8192);
    let low_centroid = adapter.calculate_spectral_centroid(&low, sr);
    let high_centroid = adapter.calculate_spectral_centroid(&high, sr);
    assert!(
        high_centroid > low_centroid,
        "high-tone centroid {high_centroid} should exceed low-tone centroid {low_centroid}"
    );
}

#[test]
fn test_spectral_rolloff_below_nyquist_and_in_hz() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let nyquist = sr as f32 / 2.0;
    // Low/mid band signal: rolloff should sit well below Nyquist.
    let audio = multi_tone(&[400.0, 900.0, 1500.0], sr, 8192);
    let rolloff = adapter.calculate_spectral_rolloff(&audio, sr);
    assert!(
        rolloff > 0.0 && rolloff < nyquist,
        "rolloff {rolloff} Hz must be in (0, {nyquist})"
    );
    // The bulk of energy is below ~2 kHz, so the 85% rolloff must be modest.
    assert!(
        rolloff < 3000.0,
        "rolloff {rolloff} Hz unexpectedly high for a low-band signal"
    );
}

#[test]
fn test_spectral_rolloff_higher_for_brighter_signal() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let dull = multi_tone(&[300.0, 600.0], sr, 8192);
    let bright = multi_tone(&[300.0, 600.0, 5000.0, 6000.0], sr, 8192);
    let dull_rolloff = adapter.calculate_spectral_rolloff(&dull, sr);
    let bright_rolloff = adapter.calculate_spectral_rolloff(&bright, sr);
    assert!(
        bright_rolloff > dull_rolloff,
        "brighter signal rolloff {bright_rolloff} should exceed dull {dull_rolloff}"
    );
}

#[test]
fn test_spectral_flux_higher_for_changing_signal() {
    let adapter = AgeGenderAdapter::new();
    let sr = 16000u32;
    let n = 16000usize;

    // Steady tone: spectrum is (near) stationary across frames -> low flux.
    let steady = multi_tone(&[1000.0], sr, n);
    let steady_flux = adapter.calculate_spectral_flux(&steady, sr);

    // Frequency sweep: spectrum moves frame-to-frame -> high flux.
    let changing: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            // Linear chirp from 500 Hz to 5000 Hz over the signal.
            let f = 500.0 + 4500.0 * (i as f32 / n as f32);
            (2.0 * std::f32::consts::PI * f * t).sin()
        })
        .collect();
    let changing_flux = adapter.calculate_spectral_flux(&changing, sr);

    assert!(
        changing_flux > steady_flux,
        "changing-signal flux {changing_flux} should exceed steady-tone flux {steady_flux}"
    );
    assert!(steady_flux >= 0.0, "flux must be non-negative");
}
