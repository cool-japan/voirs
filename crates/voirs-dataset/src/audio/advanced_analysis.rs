//! Advanced audio analysis features
//!
//! This module provides cutting-edge audio analysis techniques including
//! perceptual loudness analysis, psychoacoustic modeling improvements,
//! and advanced spectral features for enhanced dataset quality assessment.

use crate::{AudioData, Result};
use scirs2_core::Complex;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Advanced audio analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnalysisConfig {
    /// Enable perceptual loudness calculation (LUFS/EBU R128)
    pub enable_loudness: bool,
    /// Enable bark scale analysis
    pub enable_bark_scale: bool,
    /// Enable mel scale analysis
    pub enable_mel_scale: bool,
    /// Enable chroma features for harmonic analysis
    pub enable_chroma: bool,
    /// Window size for spectral analysis
    pub window_size: usize,
    /// Hop size for overlapping analysis
    pub hop_size: usize,
    /// Minimum frequency for analysis (Hz)
    pub min_frequency: f32,
    /// Maximum frequency for analysis (Hz)
    pub max_frequency: f32,
}

impl Default for AdvancedAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_loudness: true,
            enable_bark_scale: true,
            enable_mel_scale: true,
            enable_chroma: true,
            window_size: 2048,
            hop_size: 512,
            min_frequency: 20.0,
            max_frequency: 20000.0,
        }
    }
}

/// Advanced audio features extracted from audio analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAudioFeatures {
    /// Perceptual loudness in LUFS (Loudness Units relative to Full Scale)
    pub loudness_lufs: f32,
    /// Momentary loudness range (gating method)
    pub loudness_range: f32,
    /// True peak level in dBTP
    pub true_peak_dbtp: f32,
    /// Bark scale spectral features (24 bands)
    pub bark_features: Vec<f32>,
    /// Mel scale spectral features (80 bands)
    pub mel_features: Vec<f32>,
    /// Chroma features for harmonic analysis (12 bands)
    pub chroma_features: Vec<f32>,
    /// Spectral contrast features
    pub spectral_contrast: Vec<f32>,
    /// Tonnetz features for harmonic representation
    pub tonnetz_features: Vec<f32>,
    /// Temporal features (tempo, onset density)
    pub temporal_features: TemporalFeatures,
    /// Perceptual quality score based on advanced metrics
    pub perceptual_quality: f32,
}

/// Temporal audio features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Estimated tempo in BPM
    pub tempo_bpm: f32,
    /// Onset density (onsets per second)
    pub onset_density: f32,
    /// Beat tracking confidence
    pub beat_confidence: f32,
    /// Rhythmic regularity score
    pub rhythmic_regularity: f32,
}

/// Advanced audio analyzer with modern techniques
pub struct AdvancedAudioAnalyzer {
    config: AdvancedAnalysisConfig,
    bark_filters: Vec<Vec<f32>>,
    mel_filters: Vec<Vec<f32>>,
    chroma_filters: Vec<Vec<f32>>,
}

impl AdvancedAudioAnalyzer {
    /// Create new advanced audio analyzer
    pub fn new(config: AdvancedAnalysisConfig) -> Result<Self> {
        let mut analyzer = Self {
            config: config.clone(),
            bark_filters: Vec::new(),
            mel_filters: Vec::new(),
            chroma_filters: Vec::new(),
        };

        // Pre-compute filter banks
        analyzer.initialize_filter_banks()?;

        Ok(analyzer)
    }

    /// Extract advanced features from audio data
    pub fn analyze(&mut self, audio: &AudioData) -> Result<AdvancedAudioFeatures> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        // Ensure we have mono audio for analysis
        let mono_samples = if audio.channels() == 1 {
            samples.to_vec()
        } else {
            self.stereo_to_mono(samples, audio.channels() as usize)
        };

        let mut features = AdvancedAudioFeatures {
            loudness_lufs: 0.0,
            loudness_range: 0.0,
            true_peak_dbtp: 0.0,
            bark_features: Vec::new(),
            mel_features: Vec::new(),
            chroma_features: Vec::new(),
            spectral_contrast: Vec::new(),
            tonnetz_features: Vec::new(),
            temporal_features: TemporalFeatures {
                tempo_bpm: 0.0,
                onset_density: 0.0,
                beat_confidence: 0.0,
                rhythmic_regularity: 0.0,
            },
            perceptual_quality: 0.0,
        };

        // Calculate perceptual loudness (EBU R128 approximation)
        if self.config.enable_loudness {
            features.loudness_lufs = self.calculate_loudness_lufs(&mono_samples, sample_rate)?;
            features.loudness_range = self.calculate_loudness_range(&mono_samples, sample_rate)?;
            features.true_peak_dbtp = self.calculate_true_peak(&mono_samples)?;
        }

        // Extract spectral features
        let spectrogram = self.compute_spectrogram(&mono_samples, sample_rate)?;

        if self.config.enable_bark_scale {
            features.bark_features = self.extract_bark_features(&spectrogram)?;
        }

        if self.config.enable_mel_scale {
            features.mel_features = self.extract_mel_features(&spectrogram)?;
        }

        if self.config.enable_chroma {
            features.chroma_features = self.extract_chroma_features(&spectrogram, sample_rate)?;
        }

        // Calculate spectral contrast
        features.spectral_contrast = self.calculate_spectral_contrast(&spectrogram)?;

        // Calculate tonnetz features (harmonic network)
        features.tonnetz_features = self.calculate_tonnetz_features(&features.chroma_features)?;

        // Extract temporal features
        features.temporal_features = self.extract_temporal_features(&mono_samples, sample_rate)?;

        // Calculate overall perceptual quality
        features.perceptual_quality = self.calculate_perceptual_quality(&features)?;

        Ok(features)
    }

    /// Initialize filter banks for spectral analysis
    fn initialize_filter_banks(&mut self) -> Result<()> {
        let sample_rate = 22050.0; // Standard reference sample rate
        let n_fft = self.config.window_size;
        let n_freqs = n_fft / 2 + 1;

        // Initialize Bark scale filter bank (24 bands)
        if self.config.enable_bark_scale {
            self.bark_filters = self.create_bark_filterbank(n_freqs, sample_rate)?;
        }

        // Initialize Mel scale filter bank (80 bands)
        if self.config.enable_mel_scale {
            self.mel_filters = self.create_mel_filterbank(n_freqs, sample_rate)?;
        }

        // Initialize chroma filter bank (12 bands)
        if self.config.enable_chroma {
            self.chroma_filters = self.create_chroma_filterbank(n_freqs, sample_rate)?;
        }

        Ok(())
    }

    /// Create Bark scale filter bank
    fn create_bark_filterbank(&self, n_freqs: usize, sample_rate: f32) -> Result<Vec<Vec<f32>>> {
        const N_BARK_BANDS: usize = 24;
        let mut filters = Vec::with_capacity(N_BARK_BANDS);

        // Bark scale conversion: frequency to bark
        let _freq_to_bark =
            |f: f32| -> f32 { 13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan() };

        let nyquist = sample_rate / 2.0;
        let freqs: Vec<f32> = (0..n_freqs)
            .map(|i| (i as f32 * nyquist) / (n_freqs - 1) as f32)
            .collect();

        // Create triangular filters on Bark scale
        let bark_freqs: Vec<f32> = (0..=N_BARK_BANDS + 1)
            .map(|i| {
                let bark = (i as f32 * 24.0) / (N_BARK_BANDS + 1) as f32;
                // Inverse bark scale conversion
                let f = 1960.0 * (bark + 0.53) / (26.28 - bark);
                f.min(nyquist)
            })
            .collect();

        for i in 0..N_BARK_BANDS {
            let mut filter = vec![0.0; n_freqs];
            let lower = bark_freqs[i];
            let center = bark_freqs[i + 1];
            let upper = bark_freqs[i + 2];

            for (j, &freq) in freqs.iter().enumerate() {
                if freq >= lower && freq <= upper {
                    if freq <= center {
                        filter[j] = (freq - lower) / (center - lower);
                    } else {
                        filter[j] = (upper - freq) / (upper - center);
                    }
                }
            }
            filters.push(filter);
        }

        Ok(filters)
    }

    /// Create Mel scale filter bank
    fn create_mel_filterbank(&self, n_freqs: usize, sample_rate: f32) -> Result<Vec<Vec<f32>>> {
        const N_MEL_BANDS: usize = 80;
        let mut filters = Vec::with_capacity(N_MEL_BANDS);

        // Mel scale conversion
        let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };

        let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };

        let nyquist = sample_rate / 2.0;
        let freqs: Vec<f32> = (0..n_freqs)
            .map(|i| (i as f32 * nyquist) / (n_freqs - 1) as f32)
            .collect();

        let mel_min = hz_to_mel(self.config.min_frequency);
        let mel_max = hz_to_mel(self.config.max_frequency.min(nyquist));

        // Create mel frequency points
        let mel_freqs: Vec<f32> = (0..=N_MEL_BANDS + 1)
            .map(|i| {
                let mel = mel_min + (i as f32 * (mel_max - mel_min)) / (N_MEL_BANDS + 1) as f32;
                mel_to_hz(mel)
            })
            .collect();

        for i in 0..N_MEL_BANDS {
            let mut filter = vec![0.0; n_freqs];
            let lower = mel_freqs[i];
            let center = mel_freqs[i + 1];
            let upper = mel_freqs[i + 2];

            for (j, &freq) in freqs.iter().enumerate() {
                if freq >= lower && freq <= upper {
                    if freq <= center {
                        filter[j] = (freq - lower) / (center - lower);
                    } else {
                        filter[j] = (upper - freq) / (upper - center);
                    }
                }
            }
            filters.push(filter);
        }

        Ok(filters)
    }

    /// Create chroma filter bank for harmonic analysis
    fn create_chroma_filterbank(&self, n_freqs: usize, sample_rate: f32) -> Result<Vec<Vec<f32>>> {
        const N_CHROMA: usize = 12;
        let mut filters = Vec::with_capacity(N_CHROMA);

        let nyquist = sample_rate / 2.0;
        let freqs: Vec<f32> = (0..n_freqs)
            .map(|i| (i as f32 * nyquist) / (n_freqs - 1) as f32)
            .collect();

        for chroma in 0..N_CHROMA {
            let mut filter = vec![0.0; n_freqs];

            for (j, &freq) in freqs.iter().enumerate() {
                if freq > 0.0 {
                    // Calculate the MIDI note number
                    let midi_note = 69.0 + 12.0 * (freq / 440.0).log2();
                    let note_class = (midi_note.round() as i32) % 12;

                    if note_class == chroma as i32 {
                        // Gaussian window around the target chroma
                        let deviation = (midi_note - midi_note.round()).abs();
                        filter[j] = (-0.5 * (deviation / 0.5).powi(2)).exp();
                    }
                }
            }
            filters.push(filter);
        }

        Ok(filters)
    }

    /// Convert stereo to mono by averaging channels
    fn stereo_to_mono(&self, samples: &[f32], channels: usize) -> Vec<f32> {
        if channels == 1 {
            return samples.to_vec();
        }

        let mono_len = samples.len() / channels;
        let mut mono = Vec::with_capacity(mono_len);

        for i in 0..mono_len {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += samples[i * channels + c];
            }
            mono.push(sum / channels as f32);
        }

        mono
    }

    /// Calculate perceptual loudness using EBU R128 approximation
    fn calculate_loudness_lufs(&self, samples: &[f32], sample_rate: f32) -> Result<f32> {
        // Simplified EBU R128 implementation
        // Apply pre-filter (high-pass + high-frequency shelving)
        let filtered = self.apply_r128_prefilter(samples, sample_rate)?;

        // Calculate mean square with BS.1770-4 two-stage gating
        let ms = self.calculate_gated_mean_square(&filtered, sample_rate)?;

        // Convert to LUFS
        let lufs = -0.691 + 10.0 * ms.log10();

        Ok(lufs)
    }

    /// Apply the ITU-R BS.1770-4 / EBU R128 "K-weighting" pre-filter.
    ///
    /// The K-weighting frequency response standardised in ITU-R BS.1770-4
    /// (Annex 1) and adopted by EBU R128 is realised as a cascade of two
    /// second-order sections (biquads):
    ///
    /// * **Stage 1 — high-shelf (~1.5 kHz, +4 dB):** models the acoustic effect
    ///   of the head ("pre-filter"), gently boosting high frequencies.
    /// * **Stage 2 — RLB high-pass (~38 Hz):** the revised low-frequency
    ///   B-weighting curve, strongly attenuating DC and sub-bass energy.
    ///
    /// The biquad coefficients are obtained from the canonical
    /// bilinear-transform derivation shared with [`AudioData`]
    /// ([`AudioData::k_weighting_pre_coeffs`] / [`AudioData::k_weighting_rlb_coeffs`]),
    /// so the response is correct at any sample rate and reproduces the
    /// published 48 kHz reference coefficients exactly. Filtering is performed
    /// in `f64` (Direct Form II Transposed-equivalent / Direct Form II) for
    /// numerical stability, then narrowed back to `f32`.
    fn apply_r128_prefilter(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        let fs = f64::from(sample_rate);
        if fs <= 0.0 || samples.is_empty() {
            return Ok(samples.to_vec());
        }

        // Reuse the canonical BS.1770-4 K-weighting biquad coefficients derived
        // for this sample rate (do NOT re-derive / hardcode here).
        let [b0_pre, b1_pre, b2_pre, a1_pre, a2_pre] = AudioData::k_weighting_pre_coeffs(fs);
        let [b0_rlb, b1_rlb, b2_rlb, a1_rlb, a2_rlb] = AudioData::k_weighting_rlb_coeffs(fs);

        // Two biquad stages in series (Direct Form II).
        let mut w1 = [0.0f64; 2]; // state for stage 1 (high-shelf)
        let mut w2 = [0.0f64; 2]; // state for stage 2 (RLB high-pass)

        let filtered = samples
            .iter()
            .map(|&x| {
                let xd = f64::from(x);

                // Stage 1: high-shelf
                let w1n = xd - a1_pre * w1[0] - a2_pre * w1[1];
                let y1 = b0_pre * w1n + b1_pre * w1[0] + b2_pre * w1[1];
                w1[1] = w1[0];
                w1[0] = w1n;

                // Stage 2: RLB high-pass
                let w2n = y1 - a1_rlb * w2[0] - a2_rlb * w2[1];
                let y2 = b0_rlb * w2n + b1_rlb * w2[0] + b2_rlb * w2[1];
                w2[1] = w2[0];
                w2[0] = w2n;

                y2 as f32
            })
            .collect();

        Ok(filtered)
    }

    /// Calculate the gated mean square of a K-weighted signal for integrated
    /// loudness measurement, per ITU-R BS.1770-4 / EBU R128.
    ///
    /// The integrated-loudness algorithm of BS.1770-4 (§5) is built around a
    /// two-stage gating procedure applied to overlapping measurement blocks of
    /// the already K-weighted signal:
    ///
    /// 1. **Blocking:** the signal is segmented into 400 ms blocks ("gating
    ///    blocks") overlapping by 75 % (i.e. a 100 ms hop).
    /// 2. **Per-block loudness:** each block's mean square `MS` is converted to
    ///    a loudness value `L = −0.691 + 10·log10(MS)` LUFS (the `−0.691`
    ///    constant calibrates the K-weighted mean square to LKFS).
    /// 3. **Absolute gate (`Γ_a = −70 LUFS`):** blocks whose loudness falls
    ///    below −70 LUFS are discarded (they are treated as silence).
    /// 4. **Relative gate:** the mean square of the absolute-gated blocks is
    ///    converted to an integrated loudness `L_abs`; the relative threshold is
    ///    `Γ_r = L_abs − 10 LU`. Only blocks with loudness `≥ Γ_r` survive.
    /// 5. **Result:** the (linear) mean square of the relative-gated blocks is
    ///    returned. The caller converts this to LUFS with the same
    ///    `−0.691 + 10·log10(·)` mapping.
    ///
    /// Edge cases are guarded: if no block survives a gate, the algorithm falls
    /// back to the previous (looser) set — relative-gated → absolute-gated →
    /// ungated mean square — so a finite measurement is always produced.
    ///
    /// `sample_rate` (Hz) is required to size the 400 ms blocks; it is supplied
    /// by the caller, which already carries it for the K-weighting filter.
    fn calculate_gated_mean_square(&self, samples: &[f32], sample_rate: f32) -> Result<f32> {
        // Ungated mean square of the whole buffer (ultimate fallback).
        let ungated_ms = |s: &[f32]| -> f32 {
            if s.is_empty() {
                0.0
            } else {
                s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32
            }
        };

        // 400 ms gating blocks with a 100 ms hop (75 % overlap), per BS.1770-4.
        let block_size = (sample_rate as f64 * 0.4).round() as usize;
        let hop_size = (sample_rate as f64 * 0.1).round() as usize;

        // For signals shorter than a single gating block, gating is undefined;
        // fall back to the plain mean square of the available samples.
        if block_size == 0 || hop_size == 0 || samples.len() < block_size {
            return Ok(ungated_ms(samples));
        }

        // ---- (a)/(b) Per-block mean square (linear domain). ----
        // Loudness `L = −0.691 + 10·log10(MS)` is monotonic in `MS`, so the
        // gate thresholds are applied directly on mean-square values to avoid
        // redundant log/exp round-trips while remaining exactly equivalent.
        let mut block_ms: Vec<f32> = Vec::new();
        let mut start = 0usize;
        while start + block_size <= samples.len() {
            let block = &samples[start..start + block_size];
            block_ms.push(ungated_ms(block));
            start += hop_size;
        }

        if block_ms.is_empty() {
            return Ok(ungated_ms(samples));
        }

        // Absolute gate threshold Γ_a = −70 LUFS expressed as a mean square.
        // L = −0.691 + 10·log10(MS)  ⇒  MS = 10^((L + 0.691) / 10).
        let absolute_gate_lufs = -70.0_f64;
        let absolute_gate_ms = 10.0_f64.powf((absolute_gate_lufs + 0.691) / 10.0) as f32;

        // ---- (c) Absolute gating: keep blocks with loudness ≥ −70 LUFS. ----
        let abs_gated: Vec<f32> = block_ms
            .iter()
            .copied()
            .filter(|&ms| ms >= absolute_gate_ms)
            .collect();

        // If everything is below the absolute gate (effectively silence), there
        // is no meaningful loudness; fall back to the ungated mean square.
        if abs_gated.is_empty() {
            return Ok(ungated_ms(samples));
        }

        // ---- (d) Relative threshold from the absolute-gated blocks. ----
        // Integrated loudness of the absolute-gated set:
        //   L_abs = −0.691 + 10·log10(mean(MS_abs));
        // relative threshold Γ_r = L_abs − 10 LU, i.e. a factor 10^(−1) = 0.1
        // on the linear mean square.
        let mean_ms_abs = abs_gated.iter().sum::<f32>() / abs_gated.len() as f32;
        let relative_gate_ms = mean_ms_abs * 0.1; // −10 LU in the linear domain

        // ---- (e) Relative gating: keep blocks with loudness ≥ Γ_r. ----
        let rel_gated: Vec<f32> = abs_gated
            .iter()
            .copied()
            .filter(|&ms| ms >= relative_gate_ms)
            .collect();

        // ---- (f) Mean square of the final surviving set, with fallbacks. ----
        if rel_gated.is_empty() {
            // No block clears the relative gate: fall back to the absolute-gated
            // mean square (already guaranteed non-empty above).
            return Ok(mean_ms_abs);
        }

        let gated_ms = rel_gated.iter().sum::<f32>() / rel_gated.len() as f32;
        Ok(gated_ms)
    }

    /// Calculate loudness range
    fn calculate_loudness_range(&self, samples: &[f32], sample_rate: f32) -> Result<f32> {
        // Simplified implementation
        // Calculate loudness range using proper EBU R128 gating and percentile statistics
        let block_size = (sample_rate * 0.4) as usize; // 400ms blocks for gating
        let overlap = block_size / 2; // 50% overlap
        let mut block_loudness = Vec::new();

        // Calculate loudness for each block
        for start in (0..samples.len()).step_by(overlap) {
            let end = (start + block_size).min(samples.len());
            if end - start >= block_size / 2 {
                // Minimum block size
                let block = &samples[start..end];
                if let Ok(block_lufs) = self.calculate_loudness_lufs(block, sample_rate) {
                    if block_lufs > -70.0 {
                        // Above absolute threshold
                        block_loudness.push(block_lufs);
                    }
                }
            }
        }

        if block_loudness.is_empty() {
            return Ok(0.0);
        }

        // Apply relative gating (remove blocks below -10 LU relative to mean)
        let mean_loudness = block_loudness.iter().sum::<f32>() / block_loudness.len() as f32;
        let relative_threshold = mean_loudness - 10.0;
        let gated_loudness: Vec<f32> = block_loudness
            .into_iter()
            .filter(|&lufs| lufs >= relative_threshold)
            .collect();

        if gated_loudness.is_empty() {
            return Ok(0.0);
        }

        // Calculate loudness range (LRA) as 95th percentile - 10th percentile
        let mut sorted_loudness = gated_loudness;
        sorted_loudness.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_10_idx =
            ((sorted_loudness.len() as f32 * 0.10) as usize).min(sorted_loudness.len() - 1);
        let percentile_95_idx =
            ((sorted_loudness.len() as f32 * 0.95) as usize).min(sorted_loudness.len() - 1);

        let lra = sorted_loudness[percentile_95_idx] - sorted_loudness[percentile_10_idx];
        Ok(lra.max(0.0)) // LRA is always positive
    }

    /// Calculate true peak level
    fn calculate_true_peak(&self, samples: &[f32]) -> Result<f32> {
        // Find absolute peak
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // Convert to dBTP
        let db_tp = if peak > 0.0 {
            20.0 * peak.log10()
        } else {
            -100.0 // Very quiet signal
        };

        Ok(db_tp)
    }

    /// Compute spectrogram using FFT
    fn compute_spectrogram(&self, samples: &[f32], _sample_rate: f32) -> Result<Vec<Vec<f32>>> {
        let window_size = self.config.window_size;
        let hop_size = self.config.hop_size;

        let mut spectrogram = Vec::new();

        // Apply Hann window
        let window: Vec<f32> = (0..window_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (window_size - 1) as f32).cos()))
            .collect();

        for start in (0..samples.len()).step_by(hop_size) {
            if start + window_size > samples.len() {
                break;
            }

            // Windowed signal and convert to f64 complex
            let input_f64: Vec<scirs2_core::Complex<f64>> = (0..window_size)
                .map(|i| {
                    let val = samples[start + i] * window[i];
                    scirs2_core::Complex::new(val as f64, 0.0)
                })
                .collect();

            // Apply FFT
            let fft_result = scirs2_fft::fft(&input_f64, None)
                .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); window_size]);
            let buffer: Vec<Complex<f32>> = fft_result
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();

            // Calculate magnitude spectrum
            let magnitude: Vec<f32> = buffer
                .iter()
                .take(window_size / 2 + 1)
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect();

            spectrogram.push(magnitude);
        }

        Ok(spectrogram)
    }

    /// Extract Bark scale features
    fn extract_bark_features(&self, spectrogram: &[Vec<f32>]) -> Result<Vec<f32>> {
        if self.bark_filters.is_empty() || spectrogram.is_empty() {
            return Ok(vec![0.0; 24]);
        }

        let n_frames = spectrogram.len();
        let mut bark_features = vec![0.0; self.bark_filters.len()];

        for frame in spectrogram {
            for (i, filter) in self.bark_filters.iter().enumerate() {
                let mut energy = 0.0;
                for (j, &mag) in frame.iter().enumerate() {
                    if j < filter.len() {
                        energy += mag * filter[j];
                    }
                }
                bark_features[i] += energy;
            }
        }

        // Average over frames
        for feature in &mut bark_features {
            *feature /= n_frames as f32;
        }

        Ok(bark_features)
    }

    /// Extract Mel scale features
    fn extract_mel_features(&self, spectrogram: &[Vec<f32>]) -> Result<Vec<f32>> {
        if self.mel_filters.is_empty() || spectrogram.is_empty() {
            return Ok(vec![0.0; 80]);
        }

        let n_frames = spectrogram.len();
        let mut mel_features = vec![0.0; self.mel_filters.len()];

        for frame in spectrogram {
            for (i, filter) in self.mel_filters.iter().enumerate() {
                let mut energy = 0.0;
                for (j, &mag) in frame.iter().enumerate() {
                    if j < filter.len() {
                        energy += mag * filter[j];
                    }
                }
                mel_features[i] += energy;
            }
        }

        // Average over frames and apply log
        for feature in &mut mel_features {
            *feature = (*feature / n_frames as f32 + 1e-10).ln();
        }

        Ok(mel_features)
    }

    /// Extract chroma features for harmonic analysis
    fn extract_chroma_features(
        &self,
        spectrogram: &[Vec<f32>],
        _sample_rate: f32,
    ) -> Result<Vec<f32>> {
        if self.chroma_filters.is_empty() || spectrogram.is_empty() {
            return Ok(vec![0.0; 12]);
        }

        let _n_frames = spectrogram.len();
        let mut chroma_features = vec![0.0; 12];

        for frame in spectrogram {
            for (i, filter) in self.chroma_filters.iter().enumerate() {
                let mut energy = 0.0;
                for (j, &mag) in frame.iter().enumerate() {
                    if j < filter.len() {
                        energy += mag * filter[j];
                    }
                }
                chroma_features[i] += energy;
            }
        }

        // Normalize by number of frames
        let total_energy: f32 = chroma_features.iter().sum();
        if total_energy > 0.0 {
            for feature in &mut chroma_features {
                *feature /= total_energy;
            }
        }

        Ok(chroma_features)
    }

    /// Calculate spectral contrast features
    fn calculate_spectral_contrast(&self, spectrogram: &[Vec<f32>]) -> Result<Vec<f32>> {
        if spectrogram.is_empty() {
            return Ok(vec![0.0; 6]);
        }

        const N_BANDS: usize = 6;
        let mut contrast_features = vec![0.0; N_BANDS];

        // Define frequency bands for contrast calculation
        let bands = [
            (0, spectrogram[0].len() / 8),
            (spectrogram[0].len() / 8, spectrogram[0].len() / 4),
            (spectrogram[0].len() / 4, spectrogram[0].len() / 2),
            (spectrogram[0].len() / 2, 3 * spectrogram[0].len() / 4),
            (3 * spectrogram[0].len() / 4, 7 * spectrogram[0].len() / 8),
            (7 * spectrogram[0].len() / 8, spectrogram[0].len()),
        ];

        for (band_idx, &(start, end)) in bands.iter().enumerate() {
            let mut peak_energy: f32 = 0.0;
            let mut valley_energy = f32::INFINITY;

            for frame in spectrogram {
                let band_energy: f32 = frame[start..end.min(frame.len())].iter().sum();
                peak_energy = peak_energy.max(band_energy);
                valley_energy = valley_energy.min(band_energy);
            }

            // Calculate contrast as ratio of peak to valley
            if valley_energy > 0.0 {
                contrast_features[band_idx] = (peak_energy / valley_energy).log10();
            }
        }

        Ok(contrast_features)
    }

    /// Calculate tonnetz features (harmonic network representation)
    fn calculate_tonnetz_features(&self, chroma: &[f32]) -> Result<Vec<f32>> {
        if chroma.len() != 12 {
            return Ok(vec![0.0; 6]);
        }

        let mut tonnetz = vec![0.0; 6];

        // Major thirds circle
        for (i, &chroma_val) in chroma.iter().enumerate().take(12) {
            let angle = 2.0 * PI * i as f32 / 12.0 * 3.0; // Major third
            tonnetz[0] += chroma_val * angle.cos();
            tonnetz[1] += chroma_val * angle.sin();
        }

        // Minor thirds circle
        for (i, &chroma_val) in chroma.iter().enumerate().take(12) {
            let angle = 2.0 * PI * i as f32 / 12.0 * 4.0; // Minor third
            tonnetz[2] += chroma_val * angle.cos();
            tonnetz[3] += chroma_val * angle.sin();
        }

        // Perfect fifths circle
        for (i, &chroma_val) in chroma.iter().enumerate().take(12) {
            let angle = 2.0 * PI * i as f32 / 12.0 * 7.0; // Perfect fifth
            tonnetz[4] += chroma_val * angle.cos();
            tonnetz[5] += chroma_val * angle.sin();
        }

        Ok(tonnetz)
    }

    /// Extract temporal features (tempo, rhythm)
    fn extract_temporal_features(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> Result<TemporalFeatures> {
        // Simplified tempo estimation using autocorrelation
        let tempo_bpm = self.estimate_tempo(samples, sample_rate)?;

        // Onset detection using spectral differences
        let onset_density = self.calculate_onset_density(samples, sample_rate)?;

        // Calculate beat confidence based on tempo strength and consistency
        let beat_confidence = self.calculate_beat_confidence(samples, sample_rate, tempo_bpm)?;

        // Calculate rhythmic regularity based on onset pattern consistency
        let rhythmic_regularity =
            self.calculate_rhythmic_regularity(samples, sample_rate, tempo_bpm)?;

        Ok(TemporalFeatures {
            tempo_bpm,
            onset_density,
            beat_confidence,
            rhythmic_regularity,
        })
    }

    /// Estimate tempo using autocorrelation
    fn estimate_tempo(&self, samples: &[f32], sample_rate: f32) -> Result<f32> {
        // Very simplified tempo estimation
        // Real implementation would use onset detection + autocorrelation
        let window_size = (sample_rate as usize).min(samples.len());
        let autocorr = self.autocorrelation(&samples[..window_size]);

        // Find peak in typical tempo range (60-180 BPM)
        let min_period = (sample_rate / 3.0) as usize; // 180 BPM
        let max_period = sample_rate as usize; // 60 BPM

        let mut best_period = min_period;
        let mut best_correlation = 0.0;

        for (period, &correlation) in autocorr
            .iter()
            .enumerate()
            .skip(min_period)
            .take(max_period.min(autocorr.len()) - min_period)
        {
            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        let tempo_bpm = 60.0 * sample_rate / best_period as f32;
        Ok(tempo_bpm.clamp(60.0, 180.0))
    }

    /// Calculate autocorrelation
    fn autocorrelation(&self, samples: &[f32]) -> Vec<f32> {
        let len = samples.len();
        let mut autocorr = vec![0.0; len];

        for lag in 0..len {
            for i in 0..(len - lag) {
                autocorr[lag] += samples[i] * samples[i + lag];
            }
            autocorr[lag] /= (len - lag) as f32;
        }

        autocorr
    }

    /// Calculate onset density
    fn calculate_onset_density(&self, samples: &[f32], sample_rate: f32) -> Result<f32> {
        // Simplified onset detection using energy changes
        let window_size = (0.1 * sample_rate) as usize; // 100ms windows
        let mut onsets = 0;

        let mut prev_energy = 0.0;
        for chunk in samples.chunks(window_size) {
            let energy: f32 = chunk.iter().map(|&x| x * x).sum();
            let energy_ratio = if prev_energy > 0.0 {
                energy / prev_energy
            } else {
                1.0
            };

            // Simple onset detection threshold
            if energy_ratio > 1.5 {
                onsets += 1;
            }
            prev_energy = energy;
        }

        let duration = samples.len() as f32 / sample_rate;
        Ok(onsets as f32 / duration)
    }

    /// Calculate overall perceptual quality score
    fn calculate_perceptual_quality(&self, features: &AdvancedAudioFeatures) -> Result<f32> {
        let mut quality_score = 0.0;
        let mut weight_sum = 0.0;

        // Loudness quality (prefer -23 LUFS ± 3 LU)
        let loudness_quality = 1.0 - ((features.loudness_lufs + 23.0).abs() / 10.0).min(1.0);
        quality_score += loudness_quality * 0.3;
        weight_sum += 0.3;

        // True peak quality (prefer < -1 dBTP)
        let peak_quality = if features.true_peak_dbtp < -1.0 {
            1.0
        } else {
            0.5
        };
        quality_score += peak_quality * 0.2;
        weight_sum += 0.2;

        // Spectral balance quality (based on mel features distribution)
        if !features.mel_features.is_empty() {
            let spectral_std = self.calculate_std(&features.mel_features);
            let spectral_quality = (1.0 / (1.0 + spectral_std)).clamp(0.0, 1.0);
            quality_score += spectral_quality * 0.3;
            weight_sum += 0.3;
        }

        // Harmonic content quality (based on chroma features)
        if !features.chroma_features.is_empty() {
            let harmonic_energy: f32 = features.chroma_features.iter().sum();
            let harmonic_quality =
                (harmonic_energy / features.chroma_features.len() as f32).clamp(0.0, 1.0);
            quality_score += harmonic_quality * 0.2;
            weight_sum += 0.2;
        }

        if weight_sum > 0.0 {
            Ok(quality_score / weight_sum)
        } else {
            Ok(0.5) // Neutral score if no features available
        }
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance.sqrt()
    }

    /// Calculate beat confidence based on tempo strength and consistency
    fn calculate_beat_confidence(
        &self,
        samples: &[f32],
        sample_rate: f32,
        tempo_bpm: f32,
    ) -> Result<f32> {
        if samples.is_empty() || tempo_bpm <= 0.0 {
            return Ok(0.0);
        }

        // Calculate beat period in samples
        let beat_period_seconds = 60.0 / tempo_bpm;
        let beat_period_samples = (beat_period_seconds * sample_rate) as usize;

        if beat_period_samples >= samples.len() {
            return Ok(0.0);
        }

        // Use autocorrelation at the beat period to measure beat strength
        let window_size = samples.len().min(beat_period_samples * 8); // Analyze up to 8 beats
        let autocorr = self.autocorrelation(&samples[..window_size]);

        // Check autocorrelation peak at beat period
        let beat_autocorr = if beat_period_samples < autocorr.len() {
            autocorr[beat_period_samples].abs()
        } else {
            0.0
        };

        // Check autocorrelation at half-beat (syncopation)
        let half_beat_period = beat_period_samples / 2;
        let half_beat_autocorr = if half_beat_period < autocorr.len() && half_beat_period > 0 {
            autocorr[half_beat_period].abs()
        } else {
            0.0
        };

        // Calculate energy distribution around beat periods
        let num_beats = window_size / beat_period_samples;
        let mut beat_energies = Vec::new();

        for beat_idx in 0..num_beats {
            let start = beat_idx * beat_period_samples;
            let end = (start + beat_period_samples).min(samples.len());

            if end > start {
                let beat_energy: f32 = samples[start..end].iter().map(|&x| x * x).sum();
                beat_energies.push(beat_energy);
            }
        }

        // Calculate coefficient of variation for beat energies (lower = more regular)
        let beat_confidence = if beat_energies.len() > 1 {
            let mean_energy = beat_energies.iter().sum::<f32>() / beat_energies.len() as f32;
            let energy_std = self.calculate_std(&beat_energies);

            // Combine autocorrelation strength with energy regularity
            let autocorr_strength = (beat_autocorr + half_beat_autocorr * 0.5).clamp(0.0, 1.0);
            let energy_regularity = if mean_energy > 0.0 {
                1.0 / (1.0 + energy_std / mean_energy)
            } else {
                0.0
            };

            (autocorr_strength * 0.6 + energy_regularity * 0.4).clamp(0.0, 1.0)
        } else {
            beat_autocorr.clamp(0.0, 1.0)
        };

        Ok(beat_confidence)
    }

    /// Calculate rhythmic regularity based on onset pattern consistency
    fn calculate_rhythmic_regularity(
        &self,
        samples: &[f32],
        sample_rate: f32,
        tempo_bpm: f32,
    ) -> Result<f32> {
        if samples.is_empty() || tempo_bpm <= 0.0 {
            return Ok(0.0);
        }

        // Detect onsets using energy-based method
        let hop_size = 512;
        let _window_size = 1024;
        let mut onset_times = Vec::new();

        let mut prev_energy = 0.0;
        for (frame_idx, chunk) in samples.chunks(hop_size).enumerate() {
            if chunk.len() < hop_size {
                break;
            }

            // Calculate energy of current frame
            let energy: f32 = chunk.iter().map(|&x| x * x).sum();

            // Detect onset as significant energy increase
            if energy > prev_energy * 1.5 && energy > 0.01 {
                let onset_time = frame_idx as f32 * hop_size as f32 / sample_rate;
                onset_times.push(onset_time);
            }

            prev_energy = energy;
        }

        if onset_times.len() < 2 {
            return Ok(0.0);
        }

        // Calculate inter-onset intervals (IOIs)
        let mut iois: Vec<f32> = Vec::new();
        for i in 1..onset_times.len() {
            iois.push(onset_times[i] - onset_times[i - 1]);
        }

        if iois.is_empty() {
            return Ok(0.0);
        }

        // Expected IOI based on tempo
        let expected_ioi = 60.0 / tempo_bpm; // One beat
        let expected_half_ioi = expected_ioi / 2.0; // Half beat
        let expected_quarter_ioi = expected_ioi / 4.0; // Quarter beat

        // Count how many IOIs match expected rhythmic divisions
        let mut matching_iois = 0;
        let tolerance = 0.1; // 10% tolerance

        for &ioi in &iois {
            // Check if IOI matches any common rhythmic division
            let matches_beat = (ioi - expected_ioi).abs() / expected_ioi < tolerance;
            let matches_half = (ioi - expected_half_ioi).abs() / expected_half_ioi < tolerance;
            let matches_quarter =
                (ioi - expected_quarter_ioi).abs() / expected_quarter_ioi < tolerance;
            let matches_double =
                (ioi - expected_ioi * 2.0).abs() / (expected_ioi * 2.0) < tolerance;

            if matches_beat || matches_half || matches_quarter || matches_double {
                matching_iois += 1;
            }
        }

        // Calculate regularity as ratio of matching IOIs
        let basic_regularity = matching_iois as f32 / iois.len() as f32;

        // Calculate IOI variance (lower variance = more regular)
        let ioi_mean = iois.iter().sum::<f32>() / iois.len() as f32;
        let ioi_variance =
            iois.iter().map(|&x| (x - ioi_mean).powi(2)).sum::<f32>() / iois.len() as f32;
        let ioi_cv = if ioi_mean > 0.0 {
            ioi_variance.sqrt() / ioi_mean
        } else {
            1.0
        };

        // Variance regularity (lower coefficient of variation = more regular)
        let variance_regularity = 1.0 / (1.0 + ioi_cv);

        // Combine both measures
        let rhythmic_regularity =
            (basic_regularity * 0.7 + variance_regularity * 0.3).clamp(0.0, 1.0);

        Ok(rhythmic_regularity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_analyzer_creation() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_advanced_analysis() {
        let config = AdvancedAnalysisConfig::default();
        let mut analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        // Create test audio (1 second of sine wave at 440 Hz)
        let sample_rate = 22050;
        let duration = 1.0;
        let frequency = 440.0;
        let samples: Vec<f32> = (0..((sample_rate as f32 * duration) as usize))
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = AudioData::new(samples, sample_rate, 1);
        let features = analyzer.analyze(&audio);

        assert!(features.is_ok());
        let features = features.unwrap();

        // Verify feature dimensions
        if !features.bark_features.is_empty() {
            assert_eq!(features.bark_features.len(), 24);
        }
        if !features.mel_features.is_empty() {
            assert_eq!(features.mel_features.len(), 80);
        }
        if !features.chroma_features.is_empty() {
            assert_eq!(features.chroma_features.len(), 12);
        }

        // Quality score should be between 0 and 1
        assert!(features.perceptual_quality >= 0.0 && features.perceptual_quality <= 1.0);
    }

    #[test]
    fn test_loudness_calculation() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        // Test with different amplitude levels
        let sample_rate = 22050.0;
        let samples_quiet = vec![0.1; 22050]; // Quiet signal
        let samples_loud = vec![0.8; 22050]; // Loud signal

        let lufs_quiet = analyzer
            .calculate_loudness_lufs(&samples_quiet, sample_rate)
            .unwrap();
        let lufs_loud = analyzer
            .calculate_loudness_lufs(&samples_loud, sample_rate)
            .unwrap();

        // Loud signal should have higher LUFS
        assert!(lufs_loud > lufs_quiet);
    }

    #[test]
    fn test_stereo_to_mono_conversion() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        // Stereo test signal
        let stereo_samples = vec![0.5, -0.5, 0.3, -0.3, 0.1, -0.1];
        let mono = analyzer.stereo_to_mono(&stereo_samples, 2);

        assert_eq!(mono.len(), 3);
        assert_eq!(mono[0], 0.0); // (0.5 + -0.5) / 2
        assert_eq!(mono[1], 0.0); // (0.3 + -0.3) / 2
        assert_eq!(mono[2], 0.0); // (0.1 + -0.1) / 2
    }

    #[test]
    fn test_tempo_estimation() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        // Create a test signal with some rhythmic content
        let sample_rate = 22050.0;
        let samples = vec![0.5; 22050]; // Simple constant signal

        let tempo = analyzer.estimate_tempo(&samples, sample_rate).unwrap();

        // Should return a reasonable tempo value
        assert!((60.0..=180.0).contains(&tempo));
    }

    /// Build a single-frequency sine wave (deterministic) for filter tests.
    fn make_sine(freq: f32, sample_rate: f32, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * freq * t).sin() * amplitude
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// BS.1770-4 gating: a louder block must dominate the gated mean square over
    /// a quiet block. The relative gate (−10 LU) discards the quiet region, so
    /// the gated MS sits close to the loud block's MS and far above the naive
    /// ungated average that would dilute the loud energy with the quiet energy.
    #[test]
    fn test_gating_louder_block_dominates() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        let sample_rate = 48_000.0_f32;
        // 5 s loud (constant 0.8) followed by 5 s quiet (0.05). Long regions
        // keep the handful of 75 %-overlap boundary blocks negligible.
        let region = 5 * sample_rate as usize;
        let mut samples = vec![0.8_f32; region];
        samples.extend(std::iter::repeat_n(0.05_f32, region));

        let gated_ms = analyzer
            .calculate_gated_mean_square(&samples, sample_rate)
            .unwrap();

        // Naive ungated mean square would average the loud and quiet halves
        // (~0.32). The gated value must instead track the loud block.
        let ungated_ms = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let loud_ms = 0.8_f32 * 0.8_f32; // 0.64
        let quiet_ms = 0.05_f32 * 0.05_f32; // 0.0025

        assert!(
            gated_ms > ungated_ms,
            "gated MS {gated_ms} should exceed ungated MS {ungated_ms}"
        );
        assert!(
            gated_ms >= 0.9 * loud_ms,
            "gated MS {gated_ms} should track loud-block MS {loud_ms}"
        );
        // And it must be nowhere near the quiet level (quiet region gated out).
        assert!(
            gated_ms > quiet_ms * 100.0,
            "gated MS {gated_ms} must not be dragged toward quiet MS {quiet_ms}"
        );
    }

    /// BS.1770-4 absolute gate: blocks below −70 LUFS (here, exact silence) are
    /// excluded. A signal that is silence + one loud burst must yield a gated
    /// mean square close to the burst's own MS, NOT diluted by the silence.
    #[test]
    fn test_gating_excludes_silence_below_absolute_gate() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        let sample_rate = 48_000.0_f32;
        let burst_amp = 0.5_f32;
        let burst_len = 5 * sample_rate as usize; // 5 s loud burst
        let sil = 3 * sample_rate as usize; // 3 s silence each side

        // 3 s silence, then a 5 s burst, then 3 s silence.
        let mut samples = vec![0.0_f32; sil];
        samples.extend(std::iter::repeat_n(burst_amp, burst_len));
        samples.extend(std::iter::repeat_n(0.0_f32, sil));

        let gated_ms = analyzer
            .calculate_gated_mean_square(&samples, sample_rate)
            .unwrap();

        let burst_ms = burst_amp * burst_amp; // 0.25
        let ungated_ms = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;

        // The silence (loudness = −∞, well below −70 LUFS) is absolute-gated
        // out; the gated MS tracks the burst, far above the silence-diluted
        // ungated average (~0.11 over the full 11 s).
        assert!(
            gated_ms >= 0.9 * burst_ms,
            "gated MS {gated_ms} should track burst MS {burst_ms}, not diluted {ungated_ms}"
        );
        assert!(
            gated_ms > ungated_ms * 1.8,
            "gated MS {gated_ms} should far exceed silence-diluted ungated {ungated_ms}"
        );
    }

    /// All-silence input must not panic and should report ~0 mean square via the
    /// ungated fallback (no block clears the absolute gate).
    #[test]
    fn test_gating_all_silence_fallback() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        let sample_rate = 48_000.0_f32;
        let samples = vec![0.0_f32; 2 * sample_rate as usize];

        let gated_ms = analyzer
            .calculate_gated_mean_square(&samples, sample_rate)
            .unwrap();
        assert!(gated_ms.abs() < 1e-9, "all-silence gated MS should be ~0");
    }

    /// Short signals (shorter than one 400 ms gating block) fall back to a plain
    /// mean square without panicking.
    #[test]
    fn test_gating_short_signal_fallback() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        let sample_rate = 48_000.0_f32;
        let samples = vec![0.5_f32; 100]; // ~2 ms, far below 400 ms
        let gated_ms = analyzer
            .calculate_gated_mean_square(&samples, sample_rate)
            .unwrap();
        // Plain mean square of constant 0.5 is 0.25.
        assert!((gated_ms - 0.25).abs() < 1e-6);
    }

    /// The BS.1770-4 K-weighting pre-filter must strongly attenuate sub-50 Hz
    /// energy (the RLB high-pass stage), while passing mid-band energy near
    /// unity.
    #[test]
    fn test_prefilter_attenuates_sub_50hz() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();

        let sample_rate = 48_000.0_f32;
        let n = sample_rate as usize; // 1 s
                                      // Skip a startup transient when measuring steady-state RMS.
        let skip = sample_rate as usize / 10; // 100 ms

        // Sub-50 Hz tone (30 Hz): should be heavily attenuated.
        let low = make_sine(30.0, sample_rate, n, 0.5);
        let low_filt = analyzer.apply_r128_prefilter(&low, sample_rate).unwrap();
        let low_ratio = rms(&low_filt[skip..]) / rms(&low[skip..]);

        // Mid-band tone (1 kHz): should pass close to unity (high-shelf adds a
        // little gain, RLB does not attenuate here).
        let mid = make_sine(1_000.0, sample_rate, n, 0.5);
        let mid_filt = analyzer.apply_r128_prefilter(&mid, sample_rate).unwrap();
        let mid_ratio = rms(&mid_filt[skip..]) / rms(&mid[skip..]);

        // 30 Hz is well into the RLB stop-band → strong attenuation.
        assert!(
            low_ratio < 0.6,
            "30 Hz energy should be attenuated, ratio = {low_ratio}"
        );
        // Mid-band passes essentially unchanged (≈ +0..+1 dB from the shelf).
        assert!(
            (0.9..=1.3).contains(&mid_ratio),
            "1 kHz energy should pass near unity, ratio = {mid_ratio}"
        );
        // And crucially the low tone is attenuated far more than the mid tone.
        assert!(
            low_ratio < mid_ratio * 0.7,
            "sub-50 Hz ({low_ratio}) must be attenuated more than mid-band ({mid_ratio})"
        );
    }

    /// Empty input to the pre-filter must return an empty buffer, not panic.
    #[test]
    fn test_prefilter_empty_input() {
        let config = AdvancedAnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(config).unwrap();
        let out = analyzer.apply_r128_prefilter(&[], 48_000.0).unwrap();
        assert!(out.is_empty());
    }
}
