//! Tests for core voice conversion functionality

use super::converter::VoiceConverter;
use crate::types::{
    AudioSample, ConversionRequest, ConversionTarget, ConversionType, VoiceCharacteristics,
};
use std::time::Duration;

#[tokio::test]
async fn test_voice_converter_creation() {
    let converter = VoiceConverter::new();
    assert!(converter.is_ok());

    let converter = converter.unwrap();
    assert_eq!(converter.config.output_sample_rate, 22050);
}

#[tokio::test]
async fn test_voice_converter_with_config() {
    let mut config = crate::config::ConversionConfig::default();
    config.output_sample_rate = 22050;
    config.buffer_size = 512;
    config.quality_level = 0.9;
    config.use_gpu = false;

    let converter = VoiceConverter::with_config(config.clone());
    assert!(converter.is_ok());

    let converter = converter.unwrap();
    assert_eq!(converter.config.output_sample_rate, 22050);
    assert_eq!(converter.config.buffer_size, 512);
    assert_eq!(converter.config.quality_level, 0.9);
}

#[tokio::test]
async fn test_voice_converter_builder() {
    let config = crate::config::ConversionConfig::default();
    let converter = VoiceConverter::builder().config(config.clone()).build();

    assert!(converter.is_ok());
    let converter = converter.unwrap();
    assert_eq!(
        converter.config.output_sample_rate,
        config.output_sample_rate
    );
}

#[tokio::test]
async fn test_simple_pitch_conversion() {
    let converter = VoiceConverter::new().unwrap();

    // Create test audio (simple sine wave)
    let mut test_audio = Vec::new();
    for i in 0..1000 {
        let sample = (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.3;
        test_audio.push(sample);
    }

    let characteristics = VoiceCharacteristics {
        pitch: crate::types::PitchCharacteristics {
            mean_f0: 220.0, // Octave down
            ..Default::default()
        },
        ..Default::default()
    };

    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_pitch".to_string(),
        test_audio,
        44100,
        ConversionType::PitchShift,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert!(!result.converted_audio.is_empty());
    assert_eq!(result.output_sample_rate, 22050);
    assert!(result.processing_time > Duration::from_nanos(0));

    // Check that quality metrics were generated
    assert!(!result.quality_metrics.is_empty());
    assert!(result.artifacts.is_some());
    assert!(result.objective_quality.is_some());
}

#[tokio::test]
async fn test_speed_conversion() {
    let converter = VoiceConverter::new().unwrap();

    // Create test audio
    let test_audio = vec![0.1, -0.1, 0.2, -0.2, 0.1, -0.1];

    let characteristics = VoiceCharacteristics {
        timing: crate::types::TimingCharacteristics {
            speaking_rate: 1.5, // 1.5x faster
            ..Default::default()
        },
        ..Default::default()
    };

    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_speed".to_string(),
        test_audio.clone(),
        16000,
        ConversionType::SpeedTransformation,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert!(!result.converted_audio.is_empty());
}

#[tokio::test]
async fn test_age_transformation() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.15, -0.15, 0.05, -0.05];

    let characteristics = VoiceCharacteristics::for_age(crate::types::AgeGroup::Senior);
    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_age".to_string(),
        test_audio,
        22050,
        ConversionType::AgeTransformation,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert!(!result.converted_audio.is_empty());
    assert!(result.artifacts.is_some());
    assert!(result.objective_quality.is_some());
}

#[tokio::test]
async fn test_gender_transformation() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05];

    let characteristics = VoiceCharacteristics::for_gender(crate::types::Gender::Female);
    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_gender".to_string(),
        test_audio,
        44100,
        ConversionType::GenderTransformation,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert!(!result.converted_audio.is_empty());
}

#[tokio::test]
async fn test_voice_morphing_with_reference_samples() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2];

    // Create reference samples
    let reference_sample =
        AudioSample::new("ref1".to_string(), vec![0.15, -0.15, 0.25, -0.25], 16000);

    let characteristics = VoiceCharacteristics::default();
    let target = ConversionTarget::new(characteristics).with_reference_sample(reference_sample);

    let request = ConversionRequest::new(
        "test_morph".to_string(),
        test_audio,
        16000,
        ConversionType::VoiceMorphing,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_speaker_conversion_with_speaker_id() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15];

    let characteristics = VoiceCharacteristics::for_gender(crate::types::Gender::Male);
    let target = ConversionTarget::new(characteristics).with_speaker_id("speaker_123".to_string());

    let request = ConversionRequest::new(
        "test_speaker".to_string(),
        test_audio,
        22050,
        ConversionType::SpeakerConversion,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert!(!result.converted_audio.is_empty());
}

#[tokio::test]
async fn test_emotional_transformation() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2, 0.1, -0.1];

    let mut characteristics = VoiceCharacteristics::default();
    characteristics
        .custom_params
        .insert("valence".to_string(), 0.8);
    characteristics
        .custom_params
        .insert("arousal".to_string(), 0.6);

    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_emotion".to_string(),
        test_audio,
        16000,
        ConversionType::EmotionalTransformation,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert!(!result.converted_audio.is_empty());
}

#[tokio::test]
async fn test_conversion_with_strength_and_preservation() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2];

    let characteristics = VoiceCharacteristics::for_age(crate::types::AgeGroup::Child);
    let target = ConversionTarget::new(characteristics)
        .with_strength(0.7)
        .with_preservation(0.3);

    let request = ConversionRequest::new(
        "test_strength".to_string(),
        test_audio,
        44100,
        ConversionType::AgeTransformation,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    assert_eq!(result.conversion_type, ConversionType::AgeTransformation);
}

#[tokio::test]
async fn test_quality_metrics_integration() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15];

    let characteristics = VoiceCharacteristics::default();
    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_quality".to_string(),
        test_audio,
        16000,
        ConversionType::PitchShift,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);

    // Check artifact detection results
    let artifacts = result.artifacts.unwrap();
    assert!(artifacts.overall_score >= 0.0 && artifacts.overall_score <= 1.0);
    assert!(!artifacts.artifact_types.is_empty());
    assert!(artifacts.quality_assessment.overall_quality >= 0.0);

    // Check objective quality metrics
    let quality = result.objective_quality.unwrap();
    assert!(quality.overall_score >= 0.0 && quality.overall_score <= 1.0);
    assert!(quality.spectral_similarity >= 0.0 && quality.spectral_similarity <= 1.0);
    assert!(quality.temporal_consistency >= 0.0 && quality.temporal_consistency <= 1.0);
    assert!(quality.naturalness >= 0.0 && quality.naturalness <= 1.0);
    assert!(quality.perceptual_quality >= 0.0 && quality.perceptual_quality <= 1.0);
}

#[tokio::test]
async fn test_adaptive_quality_integration() {
    let converter = VoiceConverter::new().unwrap();

    // Set a high quality target to trigger adaptive adjustments
    converter.set_quality_target(0.9).await;

    // Create audio that might have quality issues
    let mut test_audio = Vec::new();
    for i in 0..500 {
        let sample = if i % 50 == 0 {
            0.8 // Add some spikes that might be detected as artifacts
        } else {
            (i as f32 * 0.01).sin() * 0.1
        };
        test_audio.push(sample);
    }

    let characteristics = VoiceCharacteristics::for_gender(crate::types::Gender::Female);
    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_adaptive".to_string(),
        test_audio,
        22050,
        ConversionType::GenderTransformation,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);

    // Check that adaptive quality system was engaged
    assert!(result.artifacts.is_some());
    assert!(result.objective_quality.is_some());

    // Get adaptive quality stats
    let stats = converter.get_adaptive_quality_stats().await;
    assert!(!stats.is_empty());
}

#[tokio::test]
async fn test_converter_statistics() {
    let converter = VoiceConverter::new().unwrap();

    let stats = converter.get_stats().await;
    assert_eq!(stats.loaded_models, 0); // No models loaded initially
    assert_eq!(stats.cached_voices, 0); // No cached voices initially
    assert!(stats.device.contains("Cpu")); // Should use CPU by default
}

#[tokio::test]
async fn test_invalid_request_validation() {
    let converter = VoiceConverter::new().unwrap();

    // Request with empty audio
    let characteristics = VoiceCharacteristics::default();
    let target = ConversionTarget::new(characteristics);

    let invalid_request = ConversionRequest::new(
        "invalid".to_string(),
        vec![], // Empty audio
        44100,
        ConversionType::PitchShift,
        target,
    );

    let result = converter.convert(invalid_request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_custom_conversion_type() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.2, -0.2];
    let characteristics = VoiceCharacteristics::default();
    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_custom".to_string(),
        test_audio,
        16000,
        ConversionType::Custom("my_custom_model".to_string()),
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);
    // Custom conversion without loaded model should return original audio
    assert!(!result.converted_audio.is_empty());
}

#[tokio::test]
async fn test_preprocess_and_postprocess() {
    let converter = VoiceConverter::new().unwrap();

    let test_audio = vec![0.1, -0.1, 0.5, -0.5, 0.8, -0.8]; // Some samples > 0.5

    // Test preprocessing (this is internal but affects the result)
    let characteristics = VoiceCharacteristics::default();
    let target = ConversionTarget::new(characteristics);

    let request = ConversionRequest::new(
        "test_process".to_string(),
        test_audio,
        22050,
        ConversionType::PitchShift,
        target,
    );

    let result = converter.convert(request).await;
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.success);

    // The processed audio should be normalized
    let max_sample = result
        .converted_audio
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    assert!(max_sample <= 1.0); // Should be normalized
}

#[tokio::test]
async fn test_realtime_conversion_validation() {
    let characteristics = VoiceCharacteristics::default();
    let target = ConversionTarget::new(characteristics);

    // Test realtime with supported conversion type
    let request = ConversionRequest::new(
        "realtime_valid".to_string(),
        vec![0.1, -0.1, 0.2, -0.2],
        44100,
        ConversionType::PitchShift,
        target.clone(),
    )
    .with_realtime(true);

    assert!(request.validate().is_ok());

    // Test realtime with unsupported conversion type
    let invalid_request = ConversionRequest::new(
        "realtime_invalid".to_string(),
        vec![0.1, -0.1, 0.2, -0.2],
        44100,
        ConversionType::VoiceMorphing, // Does not support realtime
        target,
    )
    .with_realtime(true);

    assert!(invalid_request.validate().is_err());
}

// ── Signal-processing DSP primitives (batch-17 real-DSP) ──────────────────────

/// Deterministic pure sine tone.
fn dsp_tone(freq_hz: f32, sample_rate: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
        .collect()
}

/// Compute the ratio of high-frequency to low-frequency energy of a signal,
/// using a real FFT and splitting the spectrum at its midpoint.
fn hf_lf_ratio(signal: &[f32]) -> f32 {
    use scirs2_core::Complex;
    use scirs2_fft::RealFftPlanner;

    let n = signal.len();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); n / 2 + 1];
    fft.process(signal, &mut spectrum).unwrap();

    let bins = spectrum.len();
    let mid = bins / 2;
    let lf: f32 = spectrum[..mid].iter().map(|c| c.norm_sqr()).sum();
    let hf: f32 = spectrum[mid..].iter().map(|c| c.norm_sqr()).sum();
    hf / lf.max(1e-12)
}

/// Compute the magnitude-weighted spectral centroid (in bins) of a signal using
/// a real FFT. A lower centroid means spectral energy has moved toward lower
/// frequencies — a direct indicator of downward formant motion.
fn spectral_centroid(signal: &[f32]) -> f32 {
    use scirs2_core::Complex;
    use scirs2_fft::RealFftPlanner;

    let n = signal.len();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); n / 2 + 1];
    fft.process(signal, &mut spectrum).unwrap();

    let mut weighted = 0.0_f32;
    let mut total = 0.0_f32;
    for (bin, c) in spectrum.iter().enumerate() {
        let mag = c.norm();
        weighted += bin as f32 * mag;
        total += mag;
    }
    weighted / total.max(1e-12)
}

#[tokio::test]
async fn test_spectral_tilt_changes_hf_lf_ratio() {
    let converter = VoiceConverter::new().unwrap();
    // Broadband-ish signal: sum of low + high tones so both bands have energy.
    let signal: Vec<f32> = {
        let low = dsp_tone(500.0, 16000.0, 2048);
        let high = dsp_tone(5000.0, 16000.0, 2048);
        low.iter().zip(high.iter()).map(|(&l, &h)| l + h).collect()
    };

    let base_ratio = hf_lf_ratio(&signal);

    // Positive tilt → boost highs → larger HF/LF ratio.
    let boosted = converter.adjust_spectral_tilt(&signal, 0.8).unwrap();
    let boosted_ratio = hf_lf_ratio(&boosted);

    // Negative tilt → cut highs → smaller HF/LF ratio.
    let cut = converter.adjust_spectral_tilt(&signal, -0.8).unwrap();
    let cut_ratio = hf_lf_ratio(&cut);

    assert!(
        boosted_ratio > base_ratio,
        "positive tilt must raise HF/LF ratio: base={base_ratio}, boosted={boosted_ratio}"
    );
    assert!(
        cut_ratio < base_ratio,
        "negative tilt must lower HF/LF ratio: base={base_ratio}, cut={cut_ratio}"
    );
    assert_eq!(boosted.len(), signal.len());
}

#[tokio::test]
async fn test_spectral_tilt_zero_is_identity() {
    let converter = VoiceConverter::new().unwrap();
    let signal = dsp_tone(1000.0, 16000.0, 512);
    let out = converter.adjust_spectral_tilt(&signal, 0.0).unwrap();
    assert_eq!(out, signal);
}

#[tokio::test]
async fn test_formant_shift_warps_envelope_not_just_gain() {
    let converter = VoiceConverter::new().unwrap();
    // A formant-like resonance: a band of partials centred near 1 kHz.
    let signal: Vec<f32> = {
        let mut s = vec![0.0_f32; 4096];
        for (i, v) in s.iter_mut().enumerate() {
            let t = i as f32 / 16000.0;
            // Carrier near 1 kHz amplitude-modulated to create a formant region.
            *v = (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin();
        }
        s
    };

    // Shift formants up: spectral centroid (HF/LF ratio) should rise, and it must
    // NOT be a pure gain (a gain would leave HF/LF unchanged).
    let shifted_up = converter.shift_formants(&signal, 0.5).unwrap();
    assert_eq!(shifted_up.len(), signal.len());

    let base_ratio = hf_lf_ratio(&signal);
    let up_ratio = hf_lf_ratio(&shifted_up);
    assert!(
        up_ratio > base_ratio,
        "upward formant shift must move energy to higher frequencies: base={base_ratio}, up={up_ratio}"
    );

    // Zero shift is an identity.
    let same = converter.shift_formants(&signal, 0.0).unwrap();
    assert_eq!(same, signal);
}

#[tokio::test]
async fn test_vocal_tract_length_lowers_formants() {
    let converter = VoiceConverter::new().unwrap();

    // The vocal-tract-length adjustment warps the *spectral envelope* of a
    // harmonic source. To exercise it the way speech does, build a glottal-style
    // signal: a dense harmonic series (f0 = 160 Hz) whose partials densely fill
    // DC..~7 kHz, shaped by a single resonant formant peak in the UPPER band,
    // centred at 5 kHz (well above the 4 kHz HF/LF split for a 4096-sample /
    // 16 kHz analysis). Because the harmonics are closely spaced everywhere, the
    // envelope warp has real spectral content to relocate downward — unlike a
    // sparse two-tone signal, where there is nothing below the tones to boost.
    let sample_rate = 16000.0_f32;
    let f0 = 160.0_f32;
    let formant_hz = 5000.0_f32; // resonance above the 4 kHz split …
    let formant_bw = 900.0_f32; // … with a moderately sharp bandwidth.
    let len = 4096_usize;

    // Resonant (Lorentzian) envelope weight for a partial at frequency `f`.
    let resonance = |f: f32| {
        let d = (f - formant_hz) / formant_bw;
        1.0 / (1.0 + d * d)
    };

    let signal: Vec<f32> = {
        let mut s = vec![0.0_f32; len];
        let nyquist = sample_rate / 2.0;
        // Harmonics 1..=49 → up to ~7840 Hz (just below the 8 kHz Nyquist), densely
        // spanning the band so the envelope warp has real content to relocate.
        let max_harmonic = ((nyquist - f0) / f0).floor() as usize;
        for (i, v) in s.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            let mut acc = 0.0_f32;
            for h in 1..=max_harmonic {
                let f = h as f32 * f0;
                // Mild 1/h source roll-off, then the resonant formant shaping.
                let amp = resonance(f) / (h as f32).sqrt();
                acc += amp * (2.0 * std::f32::consts::PI * f * t).sin();
            }
            *v = acc;
        }
        // Normalise to unit peak so the input occupies a clean amplitude range.
        let peak = s.iter().fold(0.0_f32, |m, &x| m.max(x.abs())).max(1e-8);
        for x in s.iter_mut() {
            *x /= peak;
        }
        s
    };

    let base_ratio = hf_lf_ratio(&signal);
    // Longer tract (factor > 1) → envelope (and its 5 kHz formant) warps DOWN by
    // 1/1.4, moving the peak to ~3.6 kHz, i.e. below the 4 kHz split. The HF band
    // therefore loses energy relative to the LF band and HF/LF must drop.
    let longer = converter.adjust_vocal_tract_length(&signal, 1.4).unwrap();
    let longer_ratio = hf_lf_ratio(&longer);

    assert_eq!(longer.len(), signal.len());
    assert!(
        longer_ratio < base_ratio,
        "a longer vocal tract must lower formant energy ratio: base={base_ratio}, longer={longer_ratio}"
    );

    // Independent confirmation of downward formant motion: the spectral centroid
    // must also decrease (energy mass moves toward lower frequencies).
    let base_centroid = spectral_centroid(&signal);
    let longer_centroid = spectral_centroid(&longer);
    assert!(
        longer_centroid < base_centroid,
        "a longer vocal tract must lower the spectral centroid: base={base_centroid}, longer={longer_centroid}"
    );

    // Unity factor is an identity.
    let same = converter.adjust_vocal_tract_length(&signal, 1.0).unwrap();
    assert_eq!(same, signal);
}
