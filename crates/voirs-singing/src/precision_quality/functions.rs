//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    pitch::PitchContour,
    types::{Expression, NoteEvent, VoiceCharacteristics},
};
use std::collections::HashMap;

use super::types_3::NaturalnessScorer;
use super::types_4::{ExpressionFeatures, PrecisionQualityAnalyzer, TimingAnalyzer};
use super::types_5::{ExpressionModel, OnsetDetector};

/// Module-level autocorrelation F0 detector shared by multiple scorers.
pub(super) fn detect_f0_autocorr_frame(frame: &[f32], sample_rate: f32) -> f32 {
    if frame.len() < 64 {
        return 0.0;
    }
    let min_p = (sample_rate / 800.0) as usize;
    let max_p = (sample_rate / 80.0) as usize;
    let (mut best_p, mut best_c) = (0, 0.0_f32);
    for p in min_p..max_p.min(frame.len() / 2) {
        let n = (frame.len() - p) as f32;
        let c: f32 = frame[..frame.len() - p]
            .iter()
            .zip(&frame[p..])
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
            / n;
        if c > best_c {
            best_c = c;
            best_p = p;
        }
    }
    if best_c > 0.3 && best_p > 0 {
        sample_rate / best_p as f32
    } else {
        0.0
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{VoiceCharacteristics, VoiceType};
    #[test]
    fn test_precision_quality_analyzer_creation() {
        let analyzer = PrecisionQualityAnalyzer::new();
        let _pitch_gen = &analyzer.pitch_generator;
    }
    #[test]
    fn test_pitch_accuracy_calculation() {
        let mut analyzer = PrecisionQualityAnalyzer::new();
        let sample_rate = 44100.0;
        let duration = 1000.0 / sample_rate;
        let mut audio = Vec::new();
        for i in 0..1000 {
            let t = i as f32 / sample_rate;
            audio.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        let pitch_contour = PitchContour {
            time_points: vec![0.1; 10],
            f0_values: vec![440.0; 10],
            confidence: vec![0.9; 10],
            voicing: vec![true; 10],
            smoothness: 0.8,
            interpolation: crate::pitch::InterpolationMethod::Cubic,
        };
        let result =
            analyzer.calculate_precision_pitch_accuracy(&audio, &pitch_contour, sample_rate);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.accuracy_percentage >= 85.0);
    }
    #[test]
    fn test_timing_accuracy_calculation() {
        let mut analyzer = PrecisionQualityAnalyzer::new();
        let sample_rate = 44100.0;
        let duration = 1.0;
        let samples = (duration * sample_rate) as usize;
        let mut audio = vec![0.0; samples];
        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            if t < 0.25 {
                *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            } else if t >= 0.25 && t < 0.5 {
                *sample = 0.5 * (2.0 * std::f32::consts::PI * 523.25 * (t - 0.25)).sin();
            }
        }
        let note_events = vec![
            NoteEvent {
                note: String::from("A"),
                octave: 4,
                frequency: 440.0,
                duration: 0.25,
                velocity: 127.0,
                vibrato: 0.1,
                lyric: Some(String::from("la")),
                phonemes: vec![String::from("l"), String::from("a")],
                expression: Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: crate::types::Articulation::Normal,
            },
            NoteEvent {
                note: String::from("C"),
                octave: 5,
                frequency: 523.25,
                duration: 0.25,
                velocity: 127.0,
                vibrato: 0.1,
                lyric: Some(String::from("la")),
                phonemes: vec![String::from("l"), String::from("a")],
                expression: Expression::Neutral,
                timing_offset: 0.25,
                breath_before: 0.0,
                legato: false,
                articulation: crate::types::Articulation::Normal,
            },
        ];
        let result =
            analyzer.calculate_precision_timing_accuracy(&audio, &note_events, sample_rate);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.accuracy_percentage >= 0.0);
    }
    #[test]
    fn test_naturalness_score_calculation() {
        let mut analyzer = PrecisionQualityAnalyzer::new();
        let audio = vec![0.5; 44100];
        let voice_characteristics = VoiceCharacteristics {
            voice_type: VoiceType::Soprano,
            range: (200.0, 1000.0),
            f0_mean: 440.0,
            f0_std: 30.0,
            vibrato_frequency: 5.5,
            vibrato_depth: 0.05,
            breath_capacity: 8.0,
            vocal_power: 0.7,
            resonance: HashMap::new(),
            timbre: HashMap::new(),
        };
        let result =
            analyzer.calculate_enhanced_naturalness_score(&audio, &voice_characteristics, 44100.0);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.mos_score >= 4.0);
        assert!(report.mos_score <= 5.0);
    }
    #[test]
    fn test_expression_recognition() {
        let mut analyzer = PrecisionQualityAnalyzer::new();
        let sample_rate = 44100.0;
        let mut audio = Vec::new();
        for i in 0..44100 {
            let t = i as f32 / sample_rate;
            let base_freq = 440.0;
            let dynamic_variation = 1.0 + 0.2 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
            let freq_variation = 1.0 + 0.05 * (2.0 * std::f32::consts::PI * 6.0 * t).sin();
            audio.push(
                0.4 * dynamic_variation
                    * (2.0 * std::f32::consts::PI * base_freq * freq_variation * t).sin(),
            );
        }
        let target_expressions = vec![Expression::Happy, Expression::Excited];
        let result = analyzer.calculate_enhanced_expression_recognition(
            &audio,
            &target_expressions,
            sample_rate,
        );
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.recognition_rate_percentage >= 0.0);
    }
    #[test]
    fn test_onset_detection() {
        let mut detector = OnsetDetector::new();
        let audio = vec![0.0; 1000];
        let result = detector.detect_onsets(&audio, 44100.0);
        assert!(result.is_ok());
    }
    #[test]
    fn test_naturalness_scorer() {
        let scorer = NaturalnessScorer::new();
        let sample_rate = 44100.0;
        let mut audio = Vec::new();
        for i in 0..1000 {
            let t = i as f32 / sample_rate;
            let base_freq = 330.0;
            let vibrato = 0.04 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
            let freq = base_freq * (1.0 + vibrato);
            audio.push(0.3 * (2.0 * std::f32::consts::PI * freq * t).sin());
        }
        let voice_characteristics = VoiceCharacteristics {
            voice_type: VoiceType::Tenor,
            range: (150.0, 600.0),
            f0_mean: 330.0,
            f0_std: 25.0,
            vibrato_frequency: 5.0,
            vibrato_depth: 0.04,
            breath_capacity: 10.0,
            vocal_power: 0.8,
            resonance: HashMap::new(),
            timbre: HashMap::new(),
        };
        let result = scorer.analyze_breath_patterns(&audio, sample_rate);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 2.5);
    }
    #[test]
    fn test_expression_model_similarity() {
        let calm_model = ExpressionModel::new_calm();
        let test_features = ExpressionFeatures {
            attack_time: 0.047,
            sustain_level: 0.82,
            decay_time: 0.19,
            spectral_centroid: 920.0,
            dynamic_range: 0.23,
        };
        let similarity = calm_model.calculate_similarity(&test_features);
        assert!(similarity > 0.6);
    }
    #[test]
    fn test_timing_analyzer() {
        let analyzer = TimingAnalyzer::new();
        let detected = vec![0.1, 0.3, 0.6];
        let target = vec![0.0, 0.25, 0.55];
        let result = analyzer.align_onsets(&detected, &target);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }
    #[test]
    fn test_energy_envelope_tracks_amplitude() {
        let scorer = NaturalnessScorer::new();
        let sr = 44100.0_f32;
        let n = 44100_usize;
        let mut audio = vec![0.0_f32; n];
        for i in (n / 2)..n {
            let t = i as f32 / sr;
            audio[i] = 0.8 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }
        let env = scorer.calculate_energy_envelope(&audio, sr).unwrap();
        assert!(!env.is_empty(), "envelope must be non-empty");
        let mid = env.len() / 2;
        let avg_lo: f32 = env[..mid].iter().sum::<f32>() / mid as f32;
        let avg_hi: f32 = env[mid..].iter().sum::<f32>() / (env.len() - mid) as f32;
        assert!(avg_hi > avg_lo * 5.0, "avg_hi={avg_hi} avg_lo={avg_lo}");
    }
    #[test]
    fn test_vibrato_rate_synthetic() {
        let scorer = NaturalnessScorer::new();
        let f0: Vec<f32> = (0..100_usize)
            .map(|i| 440.0 + 20.0 * (2.0 * std::f32::consts::PI * 5.5 * i as f32 / 50.0).sin())
            .collect();
        let rate = scorer.calculate_vibrato_rate(&f0, 44100.0).unwrap();
        assert!(
            (rate - 5.5).abs() < 1.5,
            "vibrato rate {rate} not close to 5.5 Hz"
        );
    }
    #[test]
    fn test_vibrato_depth_flat_tone_near_zero() {
        let scorer = NaturalnessScorer::new();
        let depth = scorer.calculate_vibrato_depth(&vec![440.0; 50]).unwrap();
        assert!(
            depth < 0.01,
            "flat tone vibrato depth should be near zero, got {depth}"
        );
    }
    #[test]
    fn test_formants_non_constant() {
        let scorer = NaturalnessScorer::new();
        let sr = 16000.0_f32;
        let n = 4096_usize;
        let make_tone = |f1: f32, f2: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / sr;
                    0.6 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                        + 0.2 * (2.0 * std::f32::consts::PI * f2 * t).sin()
                })
                .collect()
        };
        let fa = scorer
            .extract_formant_frequencies(&make_tone(800.0, 2400.0), sr)
            .unwrap();
        let fb = scorer
            .extract_formant_frequencies(&make_tone(300.0, 1800.0), sr)
            .unwrap();
        assert_eq!(fa.len(), 3);
        assert_eq!(fb.len(), 3);
        let differs = fa.iter().zip(fb.iter()).any(|(a, b)| (a - b).abs() > 50.0);
        assert!(
            differs,
            "different signals should produce different formants; a={fa:?} b={fb:?}"
        );
    }
}
