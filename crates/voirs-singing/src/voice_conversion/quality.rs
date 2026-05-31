use crate::Error;
use serde::{Deserialize, Serialize};

use super::SpeakerEmbedding;

/// Voice quality metrics for speaker embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityMetrics {
    /// Vocal range in semitones
    pub vocal_range: f32,
    /// Vibrato rate in Hz
    pub vibrato_rate: f32,
    /// Vibrato depth in cents
    pub vibrato_depth: f32,
    /// Breathiness factor (0.0-1.0)
    pub breathiness: f32,
    /// Roughness factor (0.0-1.0)
    pub roughness: f32,
    /// Brightness factor (0.0-1.0)
    pub brightness: f32,
}

impl Default for VoiceQualityMetrics {
    fn default() -> Self {
        Self {
            vocal_range: 24.0,
            vibrato_rate: 5.0,
            vibrato_depth: 20.0,
            breathiness: 0.3,
            roughness: 0.2,
            brightness: 0.7,
        }
    }
}

pub(super) fn calculate_quality_metrics(
    source_audio: &[f32],
    converted_audio: &[f32],
    target_speaker: &SpeakerEmbedding,
    processing_time_ms: f64,
) -> Result<super::ConversionQualityMetrics, Error> {
    let speaker_similarity = calculate_speaker_similarity(converted_audio, target_speaker)?;
    let content_preservation = calculate_content_preservation(source_audio, converted_audio)?;
    let audio_quality = calculate_audio_quality(converted_audio)?;
    let naturalness = calculate_naturalness(converted_audio)?;

    Ok(super::ConversionQualityMetrics {
        speaker_similarity,
        content_preservation,
        audio_quality,
        naturalness,
        processing_time_ms,
    })
}

pub(super) fn calculate_speaker_similarity(
    _audio: &[f32],
    _target: &SpeakerEmbedding,
) -> Result<f32, Error> {
    Ok(0.85)
}

pub(super) fn calculate_content_preservation(
    source: &[f32],
    converted: &[f32],
) -> Result<f32, Error> {
    let source_energy: f32 = source.iter().map(|x| x * x).sum();
    let converted_energy: f32 = converted.iter().map(|x| x * x).sum();

    let energy_ratio = if source_energy > 0.0 {
        (converted_energy / source_energy).min(1.0)
    } else {
        1.0
    };

    Ok(energy_ratio)
}

pub(super) fn calculate_audio_quality(audio: &[f32]) -> Result<f32, Error> {
    let max_amplitude = audio.iter().map(|x| x.abs()).fold(0.0, f32::max);
    let clipping_penalty = if max_amplitude > 0.95 { 0.5 } else { 1.0 };
    Ok(0.9 * clipping_penalty)
}

pub(super) fn calculate_naturalness(audio: &[f32]) -> Result<f32, Error> {
    let zero_crossings = audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    let naturalness_score = (zero_crossings as f32 / audio.len() as f32 * 100.0).min(1.0);
    Ok(naturalness_score)
}

/// Analyzes voice quality characteristics from audio samples using real DSP.
///
/// Computes vocal_range, vibrato_rate, vibrato_depth, breathiness, roughness,
/// and brightness via autocorrelation-based F0 tracking and spectral analysis.
pub(super) fn analyze_voice_quality(samples: &[f32]) -> Result<VoiceQualityMetrics, Error> {
    const SAMPLE_RATE: u32 = 22050;
    const FRAME_LEN: usize = 512;
    const HOP: usize = 256;

    if samples.len() < FRAME_LEN {
        return Ok(VoiceQualityMetrics::default());
    }

    // Tau range: ~80–4300 Hz at 22050 Hz
    let tau_min = FRAME_LEN / 55; // ≈ 9 samples → ~2450 Hz (upper limit)
    let tau_max = FRAME_LEN; // 512 samples → ~43 Hz (lower limit)
    let tau_min = tau_min.max(1);

    // Normalized autocorrelation closure for a single frame
    let normed_ac = |frame: &[f32], tau: usize| -> f32 {
        let n_ov = FRAME_LEN.saturating_sub(tau);
        if n_ov == 0 {
            return 0.0;
        }
        let mut cross = 0.0f32;
        let mut left_sq = 0.0f32;
        let mut right_sq = 0.0f32;
        for i in 0..n_ov {
            cross += frame[i] * frame[i + tau];
            left_sq += frame[i] * frame[i];
            right_sq += frame[i + tau] * frame[i + tau];
        }
        let denom = (left_sq * right_sq).sqrt();
        if denom > 1e-12 {
            cross / denom
        } else {
            0.0
        }
    };

    let mut voiced_f0: Vec<f32> = Vec::new();
    let mut voiced_hnr: Vec<f32> = Vec::new();

    let n = samples.len();
    let mut pos = 0;
    while pos + FRAME_LEN <= n {
        let frame = &samples[pos..pos + FRAME_LEN];

        let mean: f32 = frame.iter().sum::<f32>() / FRAME_LEN as f32;
        let centered: Vec<f32> = frame.iter().map(|&s| s - mean).collect();

        let energy: f32 = centered.iter().map(|s| s * s).sum();
        if energy < 1e-10 {
            pos += HOP;
            continue;
        }

        let mut best_corr = 0.0f32;
        let mut best_tau = tau_min;
        for tau in tau_min..=tau_max.min(FRAME_LEN - 1) {
            let r = normed_ac(&centered, tau);
            if r > best_corr {
                best_corr = r;
                best_tau = tau;
            }
        }

        if best_corr > 0.35 {
            let f0 = SAMPLE_RATE as f32 / best_tau as f32;
            voiced_f0.push(f0);
            voiced_hnr.push(best_corr);
        }

        pos += HOP;
    }

    // ── vocal_range ──────────────────────────────────────────────────────────
    let vocal_range = if voiced_f0.len() > 2 {
        let min_f0 = voiced_f0.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_f0 = voiced_f0.iter().cloned().fold(0.0f32, f32::max);
        if min_f0 > 0.0 && max_f0 > min_f0 {
            (12.0 * (max_f0 / min_f0).log2()).clamp(0.0, 48.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // ── vibrato_rate and vibrato_depth ────────────────────────────────────────
    let (vibrato_rate, vibrato_depth) = if voiced_f0.len() >= 10 {
        let n_f0 = voiced_f0.len();
        let mean_f0 = voiced_f0.iter().sum::<f32>() / n_f0 as f32;

        // Least-squares linear detrend: y = a*x + b
        let x_mean = (n_f0 - 1) as f32 / 2.0;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (i, &f) in voiced_f0.iter().enumerate() {
            let xi = i as f32 - x_mean;
            num += xi * (f - mean_f0);
            den += xi * xi;
        }
        let slope = if den > 1e-12 { num / den } else { 0.0 };
        let intercept = mean_f0 - slope * x_mean;

        let detrended: Vec<f32> = voiced_f0
            .iter()
            .enumerate()
            .map(|(i, &f)| f - (slope * i as f32 + intercept))
            .collect();

        // DFT of the detrended contour to find vibrato peak in 4–8 Hz range.
        // Spacing between frames in seconds = HOP / SAMPLE_RATE.
        let frame_rate = SAMPLE_RATE as f32 / HOP as f32;
        let nf = detrended.len();

        // Manual DFT over the 4–8 Hz band only (cheap for short contours).
        let freq_lo = 4.0f32;
        let freq_hi = 8.0f32;
        let k_lo = (freq_lo * nf as f32 / frame_rate).floor() as usize;
        let k_hi = (freq_hi * nf as f32 / frame_rate).ceil() as usize;
        let k_lo = k_lo.max(1);
        let k_hi = k_hi.min(nf / 2);

        let mut peak_mag = 0.0f32;
        let mut peak_k = k_lo;
        for k in k_lo..=k_hi {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (i, &v) in detrended.iter().enumerate() {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * i as f32 / nf as f32;
                re += v * angle.cos();
                im -= v * angle.sin();
            }
            let mag = (re * re + im * im).sqrt();
            if mag > peak_mag {
                peak_mag = mag;
                peak_k = k;
            }
        }

        let vib_rate = if k_hi >= k_lo {
            (peak_k as f32 * frame_rate / nf as f32).clamp(4.0, 8.0)
        } else {
            5.0
        };

        // Vibrato depth in cents: 100 * (peak_amplitude / mean_f0) * 2
        // peak_amplitude ≈ peak_mag / (nf/2)
        let peak_amp = peak_mag / (nf as f32 / 2.0).max(1.0);
        let vib_depth = (100.0 * (peak_amp / mean_f0.max(1.0)) * 2.0).clamp(0.0, 100.0);

        (vib_rate, vib_depth)
    } else {
        (5.0f32, 0.0f32)
    };

    // ── breathiness via HNR ──────────────────────────────────────────────────
    let breathiness = if voiced_hnr.is_empty() {
        0.5f32
    } else {
        let avg_hnr = voiced_hnr.iter().sum::<f32>() / voiced_hnr.len() as f32;
        (1.0 - avg_hnr.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    };

    // ── roughness via F0 jitter ───────────────────────────────────────────────
    let roughness = if voiced_f0.len() < 2 {
        0.0f32
    } else {
        let jitter = voiced_f0
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() / w[0].max(1.0))
            .sum::<f32>()
            / (voiced_f0.len() - 1) as f32;
        jitter.clamp(0.0, 1.0)
    };

    // ── brightness via spectral centroid ratio ────────────────────────────────
    let brightness = {
        let nyquist = SAMPLE_RATE as f32 / 2.0;
        let mut centroid_sum = 0.0f32;
        let mut frame_count = 0usize;

        let mut pos = 0;
        while pos + FRAME_LEN <= n {
            let frame = &samples[pos..pos + FRAME_LEN];

            let windowed: Vec<f64> = frame
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let w = 0.5
                        * (1.0
                            - (2.0 * std::f64::consts::PI * i as f64
                                / (FRAME_LEN - 1).max(1) as f64)
                                .cos());
                    s as f64 * w
                })
                .collect();

            let complex_in: Vec<scirs2_core::Complex<f64>> = windowed
                .iter()
                .map(|&x| scirs2_core::Complex::new(x, 0.0))
                .collect();

            if let Ok(spectrum) = scirs2_fft::rfft(&complex_in, Some(FRAME_LEN)) {
                let n_bins = spectrum.len();
                let mut weighted_sum = 0.0f64;
                let mut mag_sum = 0.0f64;
                for (k, c) in spectrum.iter().enumerate() {
                    let freq = k as f64 * SAMPLE_RATE as f64 / FRAME_LEN as f64;
                    let mag = (c.re * c.re + c.im * c.im).sqrt();
                    weighted_sum += freq * mag;
                    mag_sum += mag;
                }
                if mag_sum > 1e-12 {
                    let centroid = (weighted_sum / mag_sum) as f32;
                    centroid_sum += (centroid / nyquist).clamp(0.0, 1.0);
                    frame_count += 1;
                }
            }

            pos += HOP;
        }

        if frame_count > 0 {
            (centroid_sum / frame_count as f32).clamp(0.0, 1.0)
        } else {
            0.5f32
        }
    };

    Ok(VoiceQualityMetrics {
        vocal_range,
        vibrato_rate,
        vibrato_depth,
        breathiness,
        roughness,
        brightness,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin() * 0.8
            })
            .collect()
    }

    #[test]
    fn test_analyze_voice_quality_sine_440hz() {
        let samples = make_sine(440.0, 22050, 0.5);
        let metrics = analyze_voice_quality(&samples).expect("analyze_voice_quality must succeed");
        assert!(
            metrics.vocal_range.is_finite(),
            "vocal_range must be finite"
        );
        assert!(
            metrics.vibrato_rate.is_finite(),
            "vibrato_rate must be finite"
        );
        assert!(
            metrics.vibrato_depth.is_finite(),
            "vibrato_depth must be finite"
        );
        assert!(
            metrics.breathiness.is_finite(),
            "breathiness must be finite"
        );
        assert!(metrics.roughness.is_finite(), "roughness must be finite");
        assert!(metrics.brightness.is_finite(), "brightness must be finite");
        assert!(
            metrics.brightness > 0.0,
            "brightness must be > 0.0 for a non-silent signal, got {}",
            metrics.brightness
        );
    }

    #[test]
    fn test_analyze_voice_quality_silence() {
        let samples = vec![0.0f32; 22050];
        let metrics =
            analyze_voice_quality(&samples).expect("analyze_voice_quality must succeed on silence");
        assert_eq!(
            metrics.vocal_range, 0.0,
            "vocal_range must be 0.0 for silence (no voiced frames)"
        );
        assert!(
            metrics.breathiness.is_finite(),
            "breathiness must be finite for silence"
        );
    }

    #[test]
    fn test_analyze_voice_quality_fields_in_range() {
        let samples = make_sine(220.0, 22050, 0.5);
        let m = analyze_voice_quality(&samples).expect("analyze_voice_quality must succeed");
        assert!(
            (0.0..=48.0).contains(&m.vocal_range),
            "vocal_range out of [0,48]: {}",
            m.vocal_range
        );
        assert!(
            (0.0..=20.0).contains(&m.vibrato_rate),
            "vibrato_rate out of [0,20]: {}",
            m.vibrato_rate
        );
        assert!(
            (0.0..=100.0).contains(&m.vibrato_depth),
            "vibrato_depth out of [0,100]: {}",
            m.vibrato_depth
        );
        assert!(
            (0.0..=1.0).contains(&m.breathiness),
            "breathiness out of [0,1]: {}",
            m.breathiness
        );
        assert!(
            (0.0..=1.0).contains(&m.roughness),
            "roughness out of [0,1]: {}",
            m.roughness
        );
        assert!(
            (0.0..=1.0).contains(&m.brightness),
            "brightness out of [0,1]: {}",
            m.brightness
        );
    }
}
