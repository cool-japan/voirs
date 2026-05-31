//! Style analysis for zero-shot voice conversion

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Style analyzer for voice characteristics
pub struct StyleAnalyzer {
    /// Style extractors
    extractors: HashMap<String, Box<dyn StyleExtractor>>,

    /// Style comparators
    comparators: HashMap<String, Box<dyn StyleComparator>>,

    /// Analysis cache
    analysis_cache: Arc<RwLock<HashMap<String, StyleAnalysis>>>,

    /// Configuration
    config: StyleAnalysisConfig,
}

/// Style extractor trait
pub trait StyleExtractor: Send + Sync {
    /// Extract style features from audio
    fn extract_style(&self, audio: &[f32], sample_rate: u32) -> Result<StyleFeatures>;

    /// Get extractor name
    fn name(&self) -> &str;
}

/// Style comparator trait
pub trait StyleComparator: Send + Sync {
    /// Compare two style feature sets
    fn compare_styles(
        &self,
        style1: &StyleFeatures,
        style2: &StyleFeatures,
    ) -> Result<StyleSimilarity>;

    /// Get comparator name
    fn name(&self) -> &str;
}

/// Style features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleFeatures {
    /// Prosodic features
    pub prosodic: ProsodicStyleFeatures,

    /// Spectral features
    pub spectral: SpectralStyleFeatures,

    /// Temporal features
    pub temporal: TemporalStyleFeatures,

    /// Voice quality features
    pub voice_quality: VoiceQualityFeatures,
}

/// Prosodic style features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicStyleFeatures {
    /// Intonation patterns
    pub intonation_patterns: Vec<f32>,

    /// Rhythm characteristics
    pub rhythm_characteristics: Vec<f32>,

    /// Stress patterns
    pub stress_patterns: Vec<f32>,

    /// Pausing behavior
    pub pausing_behavior: Vec<f32>,
}

/// Spectral style features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralStyleFeatures {
    /// Formant characteristics
    pub formant_characteristics: Vec<f32>,

    /// Spectral envelope
    pub spectral_envelope: Vec<f32>,

    /// Harmonic content
    pub harmonic_content: Vec<f32>,

    /// Noise characteristics
    pub noise_characteristics: Vec<f32>,
}

/// Temporal style features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStyleFeatures {
    /// Speaking rate variations
    pub speaking_rate_variations: Vec<f32>,

    /// Articulation patterns
    pub articulation_patterns: Vec<f32>,

    /// Transition characteristics
    pub transition_characteristics: Vec<f32>,

    /// Timing precision
    pub timing_precision: Vec<f32>,
}

/// Voice quality features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityFeatures {
    /// Breathiness measures
    pub breathiness: f32,

    /// Roughness measures
    pub roughness: f32,

    /// Creakiness measures
    pub creakiness: f32,

    /// Tenseness measures
    pub tenseness: f32,

    /// Overall voice quality
    pub overall_quality: f32,
}

/// Style similarity result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleSimilarity {
    /// Overall similarity score (0.0 to 1.0)
    pub overall_similarity: f32,

    /// Prosodic similarity
    pub prosodic_similarity: f32,

    /// Spectral similarity
    pub spectral_similarity: f32,

    /// Temporal similarity
    pub temporal_similarity: f32,

    /// Voice quality similarity
    pub voice_quality_similarity: f32,

    /// Confidence score
    pub confidence: f32,
}

/// Style analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAnalysisConfig {
    /// Enable prosodic analysis
    pub enable_prosodic: bool,

    /// Enable spectral analysis
    pub enable_spectral: bool,

    /// Enable temporal analysis
    pub enable_temporal: bool,

    /// Enable voice quality analysis
    pub enable_voice_quality: bool,

    /// Analysis window size (ms)
    pub window_size: f32,

    /// Analysis hop size (ms)
    pub hop_size: f32,

    /// Feature smoothing factor
    pub smoothing_factor: f32,
}

/// Style analysis result
#[derive(Debug, Clone)]
pub struct StyleAnalysis {
    /// Extracted style features
    pub features: StyleFeatures,

    /// Analysis confidence
    pub confidence: f32,

    /// Analysis timestamp
    pub timestamp: Instant,

    /// Processing time (ms)
    pub processing_time: f32,
}

// ─── DSP helpers ────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

fn frame_rms_energies(audio: &[f32], frame_len: usize, hop: usize) -> Vec<f32> {
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    (0..num_frames)
        .map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            let slice = &audio[start..end];
            let mean_sq: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / slice.len() as f32;
            mean_sq.sqrt()
        })
        .collect()
}

fn segment_means(values: &[f32], n_segments: usize) -> Vec<f32> {
    if values.is_empty() || n_segments == 0 {
        return vec![0.0; n_segments];
    }
    (0..n_segments)
        .map(|seg| {
            let start = seg * values.len() / n_segments;
            let end = ((seg + 1) * values.len() / n_segments)
                .max(start + 1)
                .min(values.len());
            let slice = &values[start..end];
            if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f32>() / slice.len() as f32
            }
        })
        .collect()
}

fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (max - min).max(1e-10);
    v.iter().map(|&x| (x - min) / range).collect()
}

fn normalize_by_max(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(0.0_f32, f32::max).max(1e-10);
    v.iter().map(|&x| x / max).collect()
}

fn frame_mag_spectrum(frame: &[f32], window: &[f32]) -> Vec<f32> {
    let windowed: Vec<f64> = frame
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| (s * w) as f64)
        .collect();
    match scirs2_fft::rfft(&windowed, Some(windowed.len())) {
        Ok(bins) => bins.iter().map(|c| c.norm() as f32).collect(),
        Err(_) => vec![0.0; frame.len() / 2 + 1],
    }
}

fn autocorr_normalized(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let r0: f32 = frame.iter().map(|&x| x * x).sum();
    if r0 < 1e-12 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|tau| {
            let sum: f32 = (0..n.saturating_sub(tau))
                .map(|i| frame[i] * frame[i + tau])
                .sum();
            sum / r0
        })
        .collect()
}

// ─── Prosodic feature helpers ────────────────────────────────────────────────

fn compute_voiced_f0_per_frame(
    audio: &[f32],
    frame_len: usize,
    hop: usize,
    sample_rate: f32,
) -> Vec<Option<f32>> {
    let tau_min = frame_len / 6;
    let tau_max = frame_len;
    let voiced_threshold = 0.35_f32;

    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    (0..num_frames)
        .map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            let frame = &audio[start..end];
            if frame.len() < frame_len {
                return None;
            }
            let acorr = autocorr_normalized(frame);
            let best = (tau_min..tau_max.min(acorr.len()))
                .filter(|&tau| acorr[tau] > voiced_threshold)
                .max_by(|&a, &b| acorr[a].partial_cmp(&acorr[b]).unwrap());
            best.map(|tau| sample_rate / tau as f32)
        })
        .collect()
}

fn intonation_patterns(voiced_f0: &[Option<f32>]) -> Vec<f32> {
    let out: Vec<f32> = (0..10)
        .map(|seg| {
            let start = seg * voiced_f0.len() / 10;
            let end = ((seg + 1) * voiced_f0.len() / 10)
                .max(start + 1)
                .min(voiced_f0.len());
            let voiced: Vec<f32> = voiced_f0[start..end].iter().filter_map(|&x| x).collect();
            if voiced.is_empty() {
                0.0
            } else {
                (voiced.iter().sum::<f32>() / voiced.len() as f32) / 500.0
            }
        })
        .collect();
    out
}

fn rhythm_characteristics(rms: &[f32]) -> Vec<f32> {
    let means = segment_means(rms, 10);
    normalize_by_max(&means)
}

fn stress_patterns(rms: &[f32]) -> Vec<f32> {
    let seg_energy: Vec<f32> = (0..10)
        .map(|seg| {
            let start = seg * rms.len() / 10;
            let end = ((seg + 1) * rms.len() / 10).max(start + 1).min(rms.len());
            rms[start..end].iter().sum::<f32>() / (end - start).max(1) as f32
        })
        .collect();
    let ratios: Vec<f32> = (0..10)
        .map(|seg| {
            let left = if seg > 0 {
                seg_energy[seg - 1]
            } else {
                seg_energy[seg]
            };
            let right = if seg + 1 < 10 {
                seg_energy[seg + 1]
            } else {
                seg_energy[seg]
            };
            let neighbor_mean = (left + right) / 2.0;
            seg_energy[seg] / (neighbor_mean + 1e-10)
        })
        .collect();
    normalize_by_max(&ratios)
}

fn pausing_behavior(rms: &[f32]) -> Vec<f32> {
    let max_rms = rms.iter().cloned().fold(0.0_f32, f32::max);
    if max_rms < 1e-10 {
        return vec![1.0; 10];
    }
    let silence_threshold = 0.01 * max_rms;
    (0..10)
        .map(|seg| {
            let start = seg * rms.len() / 10;
            let end = ((seg + 1) * rms.len() / 10).max(start + 1).min(rms.len());
            let slice = &rms[start..end];
            if slice.is_empty() {
                return 0.0;
            }
            let silent = slice.iter().filter(|&&e| e < silence_threshold).count();
            silent as f32 / slice.len() as f32
        })
        .collect()
}

// ─── Spectral feature helpers ────────────────────────────────────────────────

fn smooth_envelope_via_cepstrum(mag: &[f32], frame_len: usize) -> Vec<f32> {
    let log_mag: Vec<f64> = mag.iter().map(|&m| (m as f64 + 1e-10).ln()).collect();
    let cepstrum = match scirs2_fft::rfft(&log_mag, Some(log_mag.len())) {
        Ok(c) => c,
        Err(_) => return mag.to_vec(),
    };
    let lifter_cutoff = (frame_len / 16).max(1);
    let mut liftered: Vec<scirs2_core::numeric::Complex64> = cepstrum.clone();
    for c in liftered.iter_mut().skip(lifter_cutoff) {
        *c = scirs2_core::numeric::Complex64::new(0.0, 0.0);
    }
    let n_real = log_mag.len();
    let n_full = (n_real - 1) * 2;
    let mut full_spec = vec![scirs2_core::numeric::Complex64::new(0.0, 0.0); n_full / 2 + 1];
    for (i, &v) in liftered.iter().enumerate() {
        if i < full_spec.len() {
            full_spec[i] = v;
        }
    }
    match scirs2_fft::irfft(&full_spec, Some(n_real)) {
        Ok(smooth_log) => smooth_log.iter().map(|&x| (x as f32).exp()).collect(),
        Err(_) => mag.to_vec(),
    }
}

fn formant_characteristics(
    audio: &[f32],
    frame_len: usize,
    hop: usize,
    sample_rate: f32,
) -> Vec<f32> {
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    let nyquist = sample_rate / 2.0;
    let hz_per_bin = nyquist / (frame_len / 2 + 1) as f32;
    let min_bin = (80.0 / hz_per_bin) as usize;
    let window = hann_window(frame_len);

    let mut accum = vec![0.0f32; 10];
    let mut frame_count = 0usize;

    for f in 0..num_frames {
        let start = f * hop;
        let end = (start + frame_len).min(audio.len());
        if end - start < frame_len {
            continue;
        }
        let frame = &audio[start..end];
        let mag = frame_mag_spectrum(frame, &window);
        let envelope = smooth_envelope_via_cepstrum(&mag, frame_len);

        let num_bins = envelope.len();
        let max_bin = num_bins.saturating_sub(1);
        let search_bins: Vec<usize> = (min_bin.min(max_bin)..=max_bin).collect();

        let mut peaks: Vec<(usize, f32)> = Vec::new();
        for &b in &search_bins {
            let left = if b > 0 { envelope[b - 1] } else { 0.0 };
            let right = if b + 1 < num_bins {
                envelope[b + 1]
            } else {
                0.0
            };
            if envelope[b] > left && envelope[b] > right {
                peaks.push((b, envelope[b]));
            }
        }
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks.truncate(5);
        peaks.sort_by_key(|p| p.0);

        while peaks.len() < 5 {
            peaks.push((0, 0.0));
        }

        let frame_vals: Vec<f32> = peaks
            .iter()
            .flat_map(|(bin, amp)| {
                let freq_norm = (*bin as f32 * hz_per_bin) / nyquist;
                let amp_norm = amp / (envelope.iter().cloned().fold(0.0_f32, f32::max).max(1e-10));
                [freq_norm.clamp(0.0, 1.0), amp_norm.clamp(0.0, 1.0)]
            })
            .collect();

        for (i, &v) in frame_vals.iter().enumerate().take(10) {
            accum[i] += v;
        }
        frame_count += 1;
    }

    if frame_count == 0 {
        return vec![0.0; 10];
    }
    accum.iter().map(|&x| x / frame_count as f32).collect()
}

fn spectral_envelope_bands(audio: &[f32]) -> Vec<f32> {
    let fft_len = audio.len().next_power_of_two();
    let padded: Vec<f64> = {
        let mut v: Vec<f64> = audio.iter().map(|&x| x as f64).collect();
        v.resize(fft_len, 0.0);
        v
    };
    let bins = match scirs2_fft::rfft(&padded, Some(fft_len)) {
        Ok(b) => b,
        Err(_) => return vec![0.0; 10],
    };
    let n_bins = bins.len();
    let log_mags: Vec<f32> = bins
        .iter()
        .map(|c| (c.norm() as f32 + 1e-10).ln())
        .collect();
    let band_means: Vec<f32> = (0..10)
        .map(|seg| {
            let start = seg * n_bins / 10;
            let end = ((seg + 1) * n_bins / 10).max(start + 1).min(n_bins);
            let slice = &log_mags[start..end];
            if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f32>() / slice.len() as f32
            }
        })
        .collect();
    normalize_by_max(
        &band_means
            .iter()
            .map(|&x| x - band_means.iter().cloned().fold(f32::INFINITY, f32::min))
            .collect::<Vec<_>>(),
    )
}

fn harmonic_content(
    audio: &[f32],
    frame_len: usize,
    hop: usize,
    voiced_f0: &[Option<f32>],
) -> Vec<f32> {
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    let tau_min = frame_len / 6;

    let hnr_per_frame: Vec<f32> = (0..num_frames)
        .map(|f| {
            if voiced_f0.get(f).and_then(|x| *x).is_none() {
                return 0.0;
            }
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            if end - start < frame_len {
                return 0.0;
            }
            let acorr = autocorr_normalized(&audio[start..end]);
            acorr[tau_min..]
                .iter()
                .cloned()
                .fold(0.0_f32, f32::max)
                .clamp(0.0, 1.0)
        })
        .collect();
    segment_means(&hnr_per_frame, 10)
}

fn noise_characteristics(audio: &[f32], frame_len: usize, hop: usize) -> Vec<f32> {
    let window = hann_window(frame_len);
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;

    let flatness_per_frame: Vec<f32> = (0..num_frames)
        .map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            if end - start < frame_len {
                return 0.0;
            }
            let mag = frame_mag_spectrum(&audio[start..end], &window);
            let log_mean = mag.iter().map(|&m| (m + 1e-10).ln()).sum::<f32>() / mag.len() as f32;
            let geo_mean = log_mean.exp();
            let arith_mean = mag.iter().sum::<f32>() / mag.len() as f32;
            (geo_mean / (arith_mean + 1e-10)).clamp(0.0, 1.0)
        })
        .collect();
    segment_means(&flatness_per_frame, 10)
}

// ─── Temporal feature helpers ────────────────────────────────────────────────

fn speaking_rate_variations(rms: &[f32]) -> Vec<f32> {
    let threshold = rms.iter().cloned().fold(0.0_f32, f32::max) * 0.05;
    let onsets: Vec<usize> = (1..rms.len())
        .filter(|&i| rms[i] > 1.5 * rms[i - 1] && rms[i] > threshold)
        .collect();

    let onset_counts: Vec<f32> = (0..10)
        .map(|seg| {
            let start = seg * rms.len() / 10;
            let end = ((seg + 1) * rms.len() / 10).max(start + 1).min(rms.len());
            onsets
                .iter()
                .filter(|&&idx| idx >= start && idx < end)
                .count() as f32
        })
        .collect();
    let max_onsets = onset_counts
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max)
        .max(1.0);
    onset_counts.iter().map(|&x| x / max_onsets).collect()
}

fn articulation_patterns(audio: &[f32], frame_len: usize, hop: usize) -> Vec<f32> {
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    let zcr_per_frame: Vec<f32> = (0..num_frames)
        .map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            let slice = &audio[start..end];
            if slice.len() < 2 {
                return 0.0;
            }
            let crossings = slice
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            crossings as f32 / slice.len() as f32
        })
        .collect();
    let means = segment_means(&zcr_per_frame, 10);
    normalize_by_max(&means)
}

fn transition_characteristics(audio: &[f32], frame_len: usize, hop: usize) -> Vec<f32> {
    let window = hann_window(frame_len);
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    let mut prev_mag: Option<Vec<f32>> = None;

    let flux_per_frame: Vec<f32> = (0..num_frames)
        .map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            if end - start < frame_len {
                prev_mag = None;
                return 0.0;
            }
            let mag = frame_mag_spectrum(&audio[start..end], &window);
            let flux = if let Some(ref prev) = prev_mag {
                let mean_mag = mag.iter().sum::<f32>() / mag.len() as f32;
                mag.iter()
                    .zip(prev.iter())
                    .map(|(&m, &p)| (m - p).max(0.0))
                    .sum::<f32>()
                    / (mean_mag + 1e-10)
            } else {
                0.0
            };
            prev_mag = Some(mag);
            flux
        })
        .collect();
    let means = segment_means(&flux_per_frame, 10);
    normalize_by_max(&means)
}

fn timing_precision(rms: &[f32]) -> Vec<f32> {
    (0..10)
        .map(|seg| {
            let start = seg * rms.len() / 10;
            let end = ((seg + 1) * rms.len() / 10).max(start + 1).min(rms.len());
            let slice = &rms[start..end];
            if slice.len() < 2 {
                return 0.0;
            }
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            if mean < 1e-10 {
                return 0.0;
            }
            let variance =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32;
            (variance.sqrt() / mean).clamp(0.0, 1.0)
        })
        .collect()
}

// ─── Voice quality helpers ───────────────────────────────────────────────────

fn compute_mean_hnr(audio: &[f32], frame_len: usize, hop: usize, voiced_f0: &[Option<f32>]) -> f32 {
    let tau_min = frame_len / 6;
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    let voiced_hnrs: Vec<f32> = (0..num_frames)
        .filter(|&f| voiced_f0.get(f).and_then(|x| *x).is_some())
        .filter_map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            if end - start < frame_len {
                return None;
            }
            let acorr = autocorr_normalized(&audio[start..end]);
            Some(
                acorr[tau_min..]
                    .iter()
                    .cloned()
                    .fold(0.0_f32, f32::max)
                    .clamp(0.0, 1.0),
            )
        })
        .collect();
    if voiced_hnrs.is_empty() {
        return 0.0;
    }
    voiced_hnrs.iter().sum::<f32>() / voiced_hnrs.len() as f32
}

fn compute_roughness(voiced_f0: &[Option<f32>]) -> f32 {
    let f0_vals: Vec<f32> = voiced_f0.iter().filter_map(|&x| x).collect();
    if f0_vals.len() < 2 {
        return 0.0;
    }
    let jitter_sum: f32 = f0_vals
        .windows(2)
        .map(|w| (w[0] - w[1]).abs() / (w[0] + 1e-10))
        .sum();
    (jitter_sum / (f0_vals.len() - 1) as f32).clamp(0.0, 1.0)
}

fn compute_creakiness(voiced_f0: &[Option<f32>]) -> f32 {
    let voiced: Vec<f32> = voiced_f0.iter().filter_map(|&x| x).collect();
    if voiced.is_empty() {
        return 0.0;
    }
    let creak_count = voiced.iter().filter(|&&f| f < 100.0).count();
    creak_count as f32 / voiced.len() as f32
}

fn compute_tenseness(
    audio: &[f32],
    frame_len: usize,
    hop: usize,
    voiced_f0: &[Option<f32>],
    sample_rate: f32,
) -> f32 {
    let nyquist = sample_rate / 2.0;
    let window = hann_window(frame_len);
    let num_frames = (audio.len().saturating_sub(frame_len)) / hop + 1;
    let hz_per_bin = nyquist / (frame_len / 2 + 1) as f32;

    let centroids: Vec<f32> = (0..num_frames)
        .filter(|&f| voiced_f0.get(f).and_then(|x| *x).is_some())
        .filter_map(|f| {
            let start = f * hop;
            let end = (start + frame_len).min(audio.len());
            if end - start < frame_len {
                return None;
            }
            let mag = frame_mag_spectrum(&audio[start..end], &window);
            let total: f32 = mag.iter().sum();
            if total < 1e-10 {
                return None;
            }
            let centroid_bin = mag
                .iter()
                .enumerate()
                .map(|(i, &m)| i as f32 * m)
                .sum::<f32>()
                / total;
            Some((centroid_bin * hz_per_bin / nyquist).clamp(0.0, 1.0))
        })
        .collect();

    if centroids.is_empty() {
        return 0.5;
    }
    centroids.iter().sum::<f32>() / centroids.len() as f32
}

// ─── Main implementation ─────────────────────────────────────────────────────

fn zero_style_features() -> StyleFeatures {
    StyleFeatures {
        prosodic: ProsodicStyleFeatures {
            intonation_patterns: vec![0.0; 10],
            rhythm_characteristics: vec![0.0; 10],
            stress_patterns: vec![0.0; 10],
            pausing_behavior: vec![1.0; 10],
        },
        spectral: SpectralStyleFeatures {
            formant_characteristics: vec![0.0; 10],
            spectral_envelope: vec![0.0; 10],
            harmonic_content: vec![0.0; 10],
            noise_characteristics: vec![0.0; 10],
        },
        temporal: TemporalStyleFeatures {
            speaking_rate_variations: vec![0.0; 10],
            articulation_patterns: vec![0.0; 10],
            transition_characteristics: vec![0.0; 10],
            timing_precision: vec![0.0; 10],
        },
        voice_quality: VoiceQualityFeatures {
            breathiness: 1.0,
            roughness: 0.5,
            creakiness: 0.0,
            tenseness: 0.5,
            overall_quality: 0.5,
        },
    }
}

impl Default for StyleAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StyleAnalyzer {
    /// Creates a new style analyzer with default configuration.
    pub fn new() -> Self {
        Self {
            extractors: HashMap::new(),
            comparators: HashMap::new(),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            config: StyleAnalysisConfig {
                enable_prosodic: true,
                enable_spectral: true,
                enable_temporal: true,
                enable_voice_quality: true,
                window_size: 25.0,
                hop_size: 10.0,
                smoothing_factor: 0.3,
            },
        }
    }

    /// Extracts comprehensive style features from audio using real DSP.
    ///
    /// Computes prosodic, spectral, temporal, and voice-quality features via
    /// frame-based analysis (512-sample frames, 256-sample hop) and rfft-based
    /// spectral processing. All output vectors are exactly length 10.
    ///
    /// # Arguments
    ///
    /// * `audio` - Input audio samples as f32 values
    /// * `sample_rate` - Audio sample rate in Hz
    ///
    /// # Returns
    ///
    /// A `Result` containing [`StyleFeatures`] or an error if extraction fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use voirs_conversion::zero_shot::style::StyleAnalyzer;
    /// let analyzer = StyleAnalyzer::new();
    /// let audio = vec![0.0f32; 16000];
    /// let style = analyzer.extract_style(&audio, 16000)?;
    /// println!("Breathiness: {}", style.voice_quality.breathiness);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn extract_style(&self, audio: &[f32], sample_rate: u32) -> Result<StyleFeatures> {
        const FRAME_LEN: usize = 512;
        const HOP: usize = 256;

        if audio.len() < FRAME_LEN {
            return Ok(zero_style_features());
        }

        let sr = sample_rate as f32;
        let rms = frame_rms_energies(audio, FRAME_LEN, HOP);
        let voiced_f0 = compute_voiced_f0_per_frame(audio, FRAME_LEN, HOP, sr);

        let intonation = intonation_patterns(&voiced_f0);
        let rhythm = rhythm_characteristics(&rms);
        let stress = stress_patterns(&rms);
        let pausing = pausing_behavior(&rms);

        let formants = formant_characteristics(audio, FRAME_LEN, HOP, sr);
        let sp_envelope = spectral_envelope_bands(audio);
        let harmonic = harmonic_content(audio, FRAME_LEN, HOP, &voiced_f0);
        let noise = noise_characteristics(audio, FRAME_LEN, HOP);

        let speaking_rate = speaking_rate_variations(&rms);
        let articulation = articulation_patterns(audio, FRAME_LEN, HOP);
        let transitions = transition_characteristics(audio, FRAME_LEN, HOP);
        let timing = timing_precision(&rms);

        let mean_hnr = compute_mean_hnr(audio, FRAME_LEN, HOP, &voiced_f0);
        let breathiness = (1.0 - mean_hnr).clamp(0.0, 1.0);
        let roughness = compute_roughness(&voiced_f0);
        let creakiness = compute_creakiness(&voiced_f0);
        let tenseness = compute_tenseness(audio, FRAME_LEN, HOP, &voiced_f0, sr);
        let overall_quality = (1.0
            - (breathiness * 0.3 + roughness * 0.3 + creakiness * 0.2 + (1.0 - tenseness) * 0.2))
            .clamp(0.0, 1.0);

        Ok(StyleFeatures {
            prosodic: ProsodicStyleFeatures {
                intonation_patterns: intonation,
                rhythm_characteristics: rhythm,
                stress_patterns: stress,
                pausing_behavior: pausing,
            },
            spectral: SpectralStyleFeatures {
                formant_characteristics: formants,
                spectral_envelope: sp_envelope,
                harmonic_content: harmonic,
                noise_characteristics: noise,
            },
            temporal: TemporalStyleFeatures {
                speaking_rate_variations: speaking_rate,
                articulation_patterns: articulation,
                transition_characteristics: transitions,
                timing_precision: timing,
            },
            voice_quality: VoiceQualityFeatures {
                breathiness,
                roughness,
                creakiness,
                tenseness,
                overall_quality,
            },
        })
    }
}

impl Default for StyleAnalysis {
    fn default() -> Self {
        Self {
            features: StyleFeatures::default(),
            confidence: 0.0,
            timestamp: Instant::now(),
            processing_time: 0.0,
        }
    }
}

impl Default for StyleFeatures {
    fn default() -> Self {
        Self {
            prosodic: ProsodicStyleFeatures::default(),
            spectral: SpectralStyleFeatures::default(),
            temporal: TemporalStyleFeatures::default(),
            voice_quality: VoiceQualityFeatures::default(),
        }
    }
}

impl Default for ProsodicStyleFeatures {
    fn default() -> Self {
        Self {
            intonation_patterns: Vec::new(),
            rhythm_characteristics: Vec::new(),
            stress_patterns: Vec::new(),
            pausing_behavior: Vec::new(),
        }
    }
}

impl Default for SpectralStyleFeatures {
    fn default() -> Self {
        Self {
            formant_characteristics: Vec::new(),
            spectral_envelope: Vec::new(),
            harmonic_content: Vec::new(),
            noise_characteristics: Vec::new(),
        }
    }
}

impl Default for TemporalStyleFeatures {
    fn default() -> Self {
        Self {
            speaking_rate_variations: Vec::new(),
            articulation_patterns: Vec::new(),
            transition_characteristics: Vec::new(),
            timing_precision: Vec::new(),
        }
    }
}

impl Default for VoiceQualityFeatures {
    fn default() -> Self {
        Self {
            breathiness: 0.0,
            roughness: 0.0,
            creakiness: 0.0,
            tenseness: 0.0,
            overall_quality: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, sample_rate: u32, duration_s: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_s) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_style_features_440hz_sine() {
        let analyzer = StyleAnalyzer::new();
        let audio = sine_wave(440.0, 22050, 0.5);
        let features = analyzer.extract_style(&audio, 22050).unwrap();
        let all_zero = features
            .spectral
            .spectral_envelope
            .iter()
            .all(|&x| x == 0.0);
        assert!(
            !all_zero,
            "spectral_envelope must not be all-zero for a 440Hz sine"
        );
        let brightness: f32 = features.spectral.spectral_envelope.iter().sum();
        assert!(brightness > 0.0, "spectral brightness must be > 0");
    }

    #[test]
    fn test_style_features_silence() {
        let analyzer = StyleAnalyzer::new();
        let audio = vec![0.0f32; 22050];
        let features = analyzer.extract_style(&audio, 22050).unwrap();
        for &v in &features.prosodic.pausing_behavior {
            assert!(
                (v - 1.0).abs() < 0.01,
                "pausing_behavior should be ~1.0 for silence, got {v}"
            );
        }
        assert!(
            features.voice_quality.breathiness > 0.9,
            "breathiness should be close to 1.0 for silence, got {}",
            features.voice_quality.breathiness
        );
    }

    #[test]
    fn test_style_features_lengths() {
        let analyzer = StyleAnalyzer::new();
        let audio = sine_wave(300.0, 16000, 1.0);
        let f = analyzer.extract_style(&audio, 16000).unwrap();
        assert_eq!(f.prosodic.intonation_patterns.len(), 10);
        assert_eq!(f.prosodic.rhythm_characteristics.len(), 10);
        assert_eq!(f.prosodic.stress_patterns.len(), 10);
        assert_eq!(f.prosodic.pausing_behavior.len(), 10);
        assert_eq!(f.spectral.formant_characteristics.len(), 10);
        assert_eq!(f.spectral.spectral_envelope.len(), 10);
        assert_eq!(f.spectral.harmonic_content.len(), 10);
        assert_eq!(f.spectral.noise_characteristics.len(), 10);
        assert_eq!(f.temporal.speaking_rate_variations.len(), 10);
        assert_eq!(f.temporal.articulation_patterns.len(), 10);
        assert_eq!(f.temporal.transition_characteristics.len(), 10);
        assert_eq!(f.temporal.timing_precision.len(), 10);
    }

    #[test]
    fn test_style_features_voiced() {
        let analyzer = StyleAnalyzer::new();
        let audio = sine_wave(220.0, 22050, 1.0);
        let features = analyzer.extract_style(&audio, 22050).unwrap();
        let all_zero = features
            .prosodic
            .intonation_patterns
            .iter()
            .all(|&x| x == 0.0);
        assert!(
            !all_zero,
            "intonation_patterns must not be all-zero for a 220Hz voiced sine"
        );
    }
}
