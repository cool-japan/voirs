//! Real, audio-grounded forced-alignment primitives.
//!
//! This module replaces the previous "mock alignment" (uniform time division with a
//! hardcoded confidence) with a genuine acoustic segmentation pipeline:
//!
//! 1. The audio is split into overlapping analysis frames; each frame's RMS energy,
//!    zero-crossing rate (ZCR) and FFT magnitude spectrum are computed
//!    ([`extract_frame_grid`]).
//! 2. A frame-to-frame "novelty" function combining spectral flux and log-energy
//!    delta is derived ([`compute_novelty`]) — a standard onset/boundary-detection
//!    signal.
//! 3. A dynamic-programming search ([`segment_via_dp`]) partitions the frames into
//!    exactly as many contiguous segments as there are reference phonemes, choosing
//!    the boundary placement that maximizes the total novelty score subject to a
//!    minimum segment length. This is the DTW/Viterbi-style refinement: instead of
//!    naive top-K peak picking (which can cluster boundaries or create zero-length
//!    segments), the DP guarantees a globally optimal, strictly increasing partition.
//! 4. Each resulting segment is scored against the acoustic profile expected for its
//!    reference phoneme (voiced/unvoiced, sonorant/obstruent energy level, spectral
//!    noisiness) to produce a real, audio-dependent Goodness-of-Pronunciation (GOP)
//!    style confidence ([`crate::pronunciation::types::evaluator_impl2`]).
//!
//! All FFTs use `scirs2_fft::rfft`, matching the established pattern in
//! `crate::audio_dsp`. No external FFT/`ndarray`/`rand` crates are used.

/// Analysis frame length, in milliseconds.
const FRAME_MS: f32 = 25.0;
/// Analysis hop length, in milliseconds.
const HOP_MS: f32 = 10.0;
/// Minimum analysis frame length, in samples (guards against degenerate FFT sizes at
/// very low sample rates).
const MIN_FRAME_LEN: usize = 32;
/// Safety cap on the number of frames considered by the DP segmentation search, to
/// bound its `O(frames * segments)` time/memory for pathologically long audio. Audio
/// longer than this (roughly 40s at the default 10ms hop) falls back to uniform
/// sample-domain segmentation instead of the boundary-evidence DP.
const MAX_DP_FRAMES: usize = 4_000;

/// Per-frame acoustic features used for boundary detection.
#[derive(Debug, Clone)]
pub(crate) struct FrameFeatures {
    /// Root-mean-square energy of the frame.
    pub(crate) rms: f32,
    /// Zero-crossing rate of the frame, in `[0, 1]`.
    pub(crate) zcr: f32,
    /// FFT magnitude spectrum (length `fft_len / 2 + 1`).
    pub(crate) spectrum: Vec<f32>,
}

/// A grid of analysis frames plus the hop length (in samples) used to produce them.
#[derive(Debug, Clone)]
pub(crate) struct FrameGrid {
    /// The extracted per-frame features, in time order.
    pub(crate) frames: Vec<FrameFeatures>,
    /// Hop length in samples between consecutive frame starts.
    pub(crate) hop_len: usize,
}

/// Hann window coefficient at index `i` of a window of length `n`.
fn hann(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos()
}

/// Root-mean-square amplitude of `samples`.
pub(crate) fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Fraction of adjacent sample pairs that cross zero, in `[0, 1]`.
pub(crate) fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// Peak short-time RMS across `samples`, using a sliding window of `window` samples
/// (50% overlap). Used to normalize segment energy against the loudest part of the
/// utterance rather than an arbitrary fixed constant.
pub(crate) fn peak_rms(samples: &[f32], window: usize) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let window = window.clamp(1, samples.len());
    let hop = (window / 2).max(1);
    let mut best = 0.0f32;
    let mut start = 0usize;
    loop {
        let end = (start + window).min(samples.len());
        let seg = rms(&samples[start..end]);
        if seg > best {
            best = seg;
        }
        if end >= samples.len() {
            break;
        }
        start += hop;
    }
    best
}

/// Normalized-autocorrelation voicing strength of `samples`, in `[0, 1]`.
///
/// This is the same normalized-autocorrelation-peak technique as
/// [`crate::audio_dsp::autocorrelation_f0`], adapted to return the continuous peak
/// correlation (voicing evidence) rather than a thresholded pitch estimate in Hz. The
/// search range is widened down to 60 Hz (vs. the 80 Hz floor used for pitch
/// estimation) so that short phoneme-level segments still have a usable lag range.
pub(crate) fn voicing_strength(samples: &[f32], sample_rate: u32) -> f32 {
    if sample_rate == 0 || samples.len() < 8 {
        return 0.0;
    }
    let min_lag = ((sample_rate as f32 / 400.0).floor() as usize).max(1);
    let max_lag_raw = (sample_rate as f32 / 60.0).ceil() as usize;
    let n = samples.len();
    if max_lag_raw <= min_lag || n <= min_lag + 1 {
        return 0.0;
    }
    let max_lag = max_lag_raw.min(n - 1);
    if max_lag <= min_lag {
        return 0.0;
    }

    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let signal: Vec<f64> = samples.iter().map(|&x| x as f64 - mean).collect();
    let energy0: f64 = signal.iter().map(|&x| x * x).sum();
    if energy0 <= 1e-12 {
        return 0.0;
    }

    let mut best = 0.0f64;
    for lag in min_lag..=max_lag {
        let mut cross = 0.0;
        for i in lag..n {
            cross += signal[i] * signal[i - lag];
        }
        let corr = cross / energy0;
        if corr > best {
            best = corr;
        }
    }
    best.clamp(0.0, 1.0) as f32
}

/// Analysis frame length in samples for `sample_rate`, matching the frame size used
/// internally by [`extract_frame_grid`]. Exposed so callers can use the same window
/// length for other per-utterance measurements (e.g. peak-RMS normalization).
pub(crate) fn frame_len_for(sample_rate: u32) -> usize {
    (((FRAME_MS / 1000.0) * sample_rate.max(1) as f32).round() as usize).max(MIN_FRAME_LEN)
}

/// Split `samples` into overlapping Hann-windowed analysis frames and compute their
/// RMS, ZCR and FFT magnitude spectrum. Returns an empty grid for empty input or a
/// zero sample rate.
pub(crate) fn extract_frame_grid(samples: &[f32], sample_rate: u32) -> FrameGrid {
    if samples.is_empty() || sample_rate == 0 {
        return FrameGrid {
            frames: Vec::new(),
            hop_len: 1,
        };
    }
    let frame_len = frame_len_for(sample_rate);
    let hop_len = (((HOP_MS / 1000.0) * sample_rate as f32).round() as usize).clamp(1, frame_len);
    let fft_len = frame_len.next_power_of_two().max(64);
    let num_bins = fft_len / 2 + 1;

    let mut frames = Vec::new();
    let mut start = 0usize;
    loop {
        let available = (samples.len() - start).min(frame_len);
        let segment = &samples[start..start + available];
        let seg_rms = rms(segment);
        let seg_zcr = zero_crossing_rate(segment);

        let mut buffer = vec![0.0f64; fft_len];
        for (i, slot) in buffer.iter_mut().enumerate().take(available) {
            *slot = segment[i] as f64 * hann(i, frame_len);
        }
        let spectrum = match scirs2_fft::rfft(&buffer, Some(fft_len)) {
            Ok(spec) => spec
                .iter()
                .take(num_bins)
                .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f32)
                .collect(),
            Err(_) => vec![0.0; num_bins],
        };
        frames.push(FrameFeatures {
            rms: seg_rms,
            zcr: seg_zcr,
            spectrum,
        });

        if available < frame_len {
            break;
        }
        start += hop_len;
        if start >= samples.len() {
            break;
        }
    }
    FrameGrid { frames, hop_len }
}

/// Frame-to-frame novelty (boundary evidence) function: a weighted combination of
/// positive spectral flux (`Σ max(0, |X_i[k]| - |X_{i-1}[k]|)`) and absolute
/// log-energy delta, each normalized to `[0, 1]` by its maximum over the utterance.
/// `novelty[0]` is always `0.0` (no preceding frame to compare against).
pub(crate) fn compute_novelty(frames: &[FrameFeatures]) -> Vec<f64> {
    let n = frames.len();
    let mut flux = vec![0.0f64; n];
    let mut energy_delta = vec![0.0f64; n];
    for i in 1..n {
        let len = frames[i].spectrum.len().min(frames[i - 1].spectrum.len());
        let mut f = 0.0f64;
        for k in 0..len {
            let d = frames[i].spectrum[k] as f64 - frames[i - 1].spectrum[k] as f64;
            if d > 0.0 {
                f += d;
            }
        }
        flux[i] = f;
        let e0 = (frames[i - 1].rms as f64 + 1e-6).ln();
        let e1 = (frames[i].rms as f64 + 1e-6).ln();
        energy_delta[i] = (e1 - e0).abs();
    }
    let max_flux = flux.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
    let max_energy = energy_delta.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
    (0..n)
        .map(|i| 0.6 * (flux[i] / max_flux) + 0.4 * (energy_delta[i] / max_energy))
        .collect()
}

/// Partition `novelty.len()` frames into exactly `num_segments` contiguous,
/// non-overlapping segments (each of at least `min_len` frames) so as to maximize the
/// total novelty score at the chosen interior boundaries.
///
/// This is a dynamic-programming (Viterbi-style) optimal segmentation: `dp[j][i]` is
/// the best achievable total boundary evidence when the first `i` frames are split
/// into `j` segments with the `j`-th segment ending exactly at frame `i`. Transitions
/// run in amortized `O(1)` via a monotonically advancing prefix-maximum pointer, for
/// overall `O(num_segments * num_frames)` time.
///
/// Returns `None` when there are not enough frames to satisfy `num_segments *
/// min_len`, or when `novelty.len()` exceeds [`MAX_DP_FRAMES`] (the caller should fall
/// back to uniform segmentation in that case). The returned vector always has
/// `num_segments + 1` entries: `[0, b_1, b_2, ..., num_frames]`.
pub(crate) fn segment_via_dp(
    novelty: &[f64],
    num_segments: usize,
    min_len: usize,
) -> Option<Vec<usize>> {
    let num_frames = novelty.len();
    if num_segments == 0 {
        return Some(vec![0]);
    }
    if num_frames > MAX_DP_FRAMES {
        return None;
    }
    let min_len = min_len.max(1);
    if num_frames < num_segments * min_len {
        return None;
    }

    const NEG_INF: f64 = f64::NEG_INFINITY;
    let mut dp = vec![vec![NEG_INF; num_frames + 1]; num_segments + 1];
    let mut back = vec![vec![0usize; num_frames + 1]; num_segments + 1];
    dp[0][0] = 0.0;

    for j in 1..=num_segments {
        let mut best_prev = NEG_INF;
        let mut best_prev_idx = 0usize;
        let mut frontier = (j - 1) * min_len;
        let start_i = j * min_len;
        for i in start_i..=num_frames {
            let allowed_upper = i - min_len;
            while frontier <= allowed_upper {
                if dp[j - 1][frontier] > best_prev {
                    best_prev = dp[j - 1][frontier];
                    best_prev_idx = frontier;
                }
                frontier += 1;
            }
            if best_prev > NEG_INF {
                let score = if j < num_segments {
                    novelty.get(i).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                let candidate = best_prev + score;
                if candidate > dp[j][i] {
                    dp[j][i] = candidate;
                    back[j][i] = best_prev_idx;
                }
            }
        }
    }

    if dp[num_segments][num_frames] <= NEG_INF {
        return None;
    }
    let mut boundaries = vec![num_frames];
    let mut j = num_segments;
    let mut i = num_frames;
    while j > 0 {
        let prev = back[j][i];
        boundaries.push(prev);
        i = prev;
        j -= 1;
    }
    boundaries.reverse();
    Some(boundaries)
}

/// Return the [`AlignedPhoneme`](voirs_recognizer::traits::AlignedPhoneme)s belonging
/// to word `word_index` (0-based) out of `total_words`.
///
/// Prefers the aligner's own `word_alignments` when they are present and the word
/// text matches (case-insensitively) the requested word. Falls back to a
/// proportional split of the flat phoneme sequence otherwise — e.g. for a hand-built
/// `PhonemeAlignment` that never populated word-level boundaries. The primary aligner
/// ([`super::evaluator_impl2::PronunciationEvaluatorImpl::align_phonemes_to_audio`])
/// always populates exact word boundaries, so the fallback only matters for alignments
/// constructed outside that path.
pub(crate) fn word_phoneme_span(
    alignment: &voirs_recognizer::traits::PhonemeAlignment,
    word_index: usize,
    word: &str,
    total_words: usize,
) -> Vec<voirs_recognizer::traits::AlignedPhoneme> {
    if let Some(word_alignment) = alignment.word_alignments.get(word_index) {
        if !word_alignment.phonemes.is_empty() && word_alignment.word.eq_ignore_ascii_case(word) {
            return word_alignment.phonemes.clone();
        }
    }
    if alignment.phonemes.is_empty() || total_words == 0 {
        return Vec::new();
    }
    let total_words = total_words.max(1);
    let chunk = alignment.phonemes.len() as f64 / total_words as f64;
    let start = ((word_index as f64) * chunk).round() as usize;
    let end = (((word_index + 1) as f64) * chunk).round() as usize;
    let start = start.min(alignment.phonemes.len());
    let end = end.clamp(start, alignment.phonemes.len());
    alignment.phonemes[start..end].to_vec()
}

/// Evenly divide `num_samples` samples into `num_segments` contiguous chunks,
/// returning `num_segments + 1` sample-index boundaries. Used when there is not
/// enough acoustic resolution (too few frames, or a degenerate/flat novelty signal)
/// to run the boundary-evidence DP.
pub(crate) fn uniform_sample_boundaries(num_samples: usize, num_segments: usize) -> Vec<usize> {
    if num_segments == 0 {
        return vec![0];
    }
    (0..=num_segments)
        .map(|i| ((i as f64 * num_samples as f64) / num_segments as f64).round() as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, len: usize, sample_rate: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect()
    }

    fn lcg_noise(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let unit = (state >> 33) as f32 / (1u64 << 31) as f32;
                2.0 * unit - 1.0
            })
            .collect()
    }

    #[test]
    fn test_rms_and_zcr_basic() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[1.0, -1.0, 1.0, -1.0]), 1.0);
        assert_eq!(zero_crossing_rate(&[1.0, 1.0, 1.0]), 0.0);
        let alternating = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert!(zero_crossing_rate(&alternating) > 0.9);
    }

    #[test]
    fn test_voicing_strength_sine_vs_noise() {
        let sr = 16_000;
        let voiced = sine(150.0, 2_000, sr);
        let noise = lcg_noise(0xABCD, 2_000);
        let v_voiced = voicing_strength(&voiced, sr);
        let v_noise = voicing_strength(&noise, sr);
        assert!(
            v_voiced > v_noise,
            "sine voicing ({v_voiced}) should exceed noise voicing ({v_noise})"
        );
        assert!(v_voiced > 0.5, "pure tone should be strongly voiced: {v_voiced}");
    }

    #[test]
    fn test_extract_frame_grid_nonempty_for_real_audio() {
        let sr = 16_000;
        let audio = sine(200.0, 8_000, sr);
        let grid = extract_frame_grid(&audio, sr);
        assert!(!grid.frames.is_empty());
        for frame in &grid.frames {
            assert!(frame.rms.is_finite());
            assert!(frame.zcr.is_finite());
            assert!(frame.spectrum.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_extract_frame_grid_empty_input() {
        let grid = extract_frame_grid(&[], 16_000);
        assert!(grid.frames.is_empty());
        let grid = extract_frame_grid(&[0.1, 0.2], 0);
        assert!(grid.frames.is_empty());
    }

    #[test]
    fn test_compute_novelty_detects_onset() {
        let sr = 16_000;
        // Silence followed by a tone: a real spectral/energy discontinuity partway
        // through the signal.
        let mut audio = vec![0.0f32; 4_000];
        audio.extend(sine(300.0, 4_000, sr));
        let grid = extract_frame_grid(&audio, sr);
        let novelty = compute_novelty(&grid.frames);
        assert_eq!(novelty.len(), grid.frames.len());
        // There should be a clear peak near the silence/tone boundary (~frame 25 at a
        // 10ms hop), well above the novelty at the very start of the (silent) signal.
        let peak = novelty
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        assert!(peak > novelty[1].max(1e-9) * 1.5, "expected a clear onset peak, got peak={peak}, novelty[1]={}", novelty[1]);
    }

    #[test]
    fn test_segment_via_dp_respects_segment_count_and_order() {
        let novelty = vec![0.0, 0.1, 0.9, 0.2, 0.8, 0.1, 0.3, 0.05, 0.6, 0.1];
        let boundaries = segment_via_dp(&novelty, 3, 1).expect("should find a feasible partition");
        assert_eq!(boundaries.len(), 4);
        assert_eq!(boundaries[0], 0);
        assert_eq!(*boundaries.last().unwrap(), novelty.len());
        for w in boundaries.windows(2) {
            assert!(w[1] > w[0], "boundaries must be strictly increasing: {boundaries:?}");
        }
    }

    #[test]
    fn test_segment_via_dp_infeasible_returns_none() {
        let novelty = vec![0.0, 0.1];
        assert!(segment_via_dp(&novelty, 5, 2).is_none());
    }

    #[test]
    fn test_segment_via_dp_prefers_high_novelty_boundaries() {
        // A single, unambiguous spike at frame 5 out of 10: with 2 segments the DP
        // should cut exactly there.
        let mut novelty = vec![0.0; 10];
        novelty[5] = 1.0;
        let boundaries = segment_via_dp(&novelty, 2, 1).expect("feasible");
        assert_eq!(boundaries, vec![0, 5, 10]);
    }

    #[test]
    fn test_word_phoneme_span_fallback_proportional_split() {
        use voirs_recognizer::traits::{AlignedPhoneme, PhonemeAlignment};

        let make_phoneme = |sym: &str| voirs_sdk::Phoneme {
            symbol: sym.to_string(),
            ipa_symbol: sym.to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
            duration_ms: Some(100.0),
            confidence: 0.9,
        };
        let phonemes: Vec<AlignedPhoneme> = (0..6)
            .map(|i| AlignedPhoneme {
                phoneme: make_phoneme(&format!("p{i}")),
                start_time: i as f32 * 0.1,
                end_time: (i + 1) as f32 * 0.1,
                confidence: 0.5,
            })
            .collect();
        let alignment = PhonemeAlignment {
            phonemes,
            total_duration: 0.6,
            alignment_confidence: 0.5,
            word_alignments: Vec::new(),
        };
        // No word_alignments populated: falls back to proportional split across 2
        // words -> 3 phonemes each.
        let first = word_phoneme_span(&alignment, 0, "hello", 2);
        let second = word_phoneme_span(&alignment, 1, "world", 2);
        assert_eq!(first.len(), 3);
        assert_eq!(second.len(), 3);
        assert_eq!(first[0].phoneme.symbol, "p0");
        assert_eq!(second[0].phoneme.symbol, "p3");
    }

    #[test]
    fn test_frame_len_for_scales_with_sample_rate() {
        assert!(frame_len_for(16_000) > frame_len_for(8_000));
        assert!(frame_len_for(0) >= MIN_FRAME_LEN);
    }

    #[test]
    fn test_uniform_sample_boundaries() {
        let boundaries = uniform_sample_boundaries(100, 4);
        assert_eq!(boundaries, vec![0, 25, 50, 75, 100]);
        assert_eq!(uniform_sample_boundaries(10, 0), vec![0]);
    }

    #[test]
    fn test_peak_rms_tracks_loudest_region() {
        let mut samples = vec![0.0f32; 1000];
        for (i, s) in samples.iter_mut().enumerate().skip(400).take(200) {
            *s = (i as f32 * 0.5).sin();
        }
        let peak = peak_rms(&samples, 100);
        assert!(peak > 0.1, "expected the loud region to dominate peak RMS: {peak}");
    }
}
