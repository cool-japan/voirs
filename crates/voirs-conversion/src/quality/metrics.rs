//! Quality metrics system for objective quality measurement and assessment

use crate::{Error, Result};
use tracing::{debug, info};

/// Default sample rate used when no explicit sample rate is available (Hz).
const DEFAULT_SAMPLE_RATE: f32 = 22050.0;
/// Minimum F0 for voiced detection (Hz) — below this is considered unvoiced/bass floor.
const F0_MIN_HZ: f32 = 80.0;
/// Maximum F0 for voiced detection (Hz) — above this is considered unvoiced/noise.
const F0_MAX_HZ: f32 = 800.0;
/// Minimum normalised autocorrelation value to declare a frame voiced.
const VOICED_THRESHOLD: f32 = 0.35;

// ---------------------------------------------------------------------------
// Psychoacoustic (Bark critical-band) model constants
// ---------------------------------------------------------------------------

/// FFT size used by `calculate_power_spectrum`; shared so the Bark bin → frequency
/// mapping in the loudness/sharpness models stays exactly in step with the spectrum.
const POWER_SPECTRUM_N_FFT: usize = 512;
/// Number of Bark critical bands spanning the audible range (~0–24 Bark).
const NUM_BARK_BANDS: usize = 24;
/// Width of each critical band on the Bark scale (Bark).
const BARK_BAND_WIDTH: f32 = 1.0;
/// Reference excitation `E0` for the specific-loudness power law, expressed in the FFT's
/// native power units for normalised `[-1, 1]` float audio.
const LOUDNESS_E0: f32 = 1.0;
/// Excitation at threshold in quiet — gates numerically-silent bands to zero loudness.
const LOUDNESS_E_THRESHOLD: f32 = 0.01;
/// Stevens' compressive power-law exponent for the intensity → loudness mapping.
const LOUDNESS_EXPONENT: f32 = 0.23;
/// Half-saturation constant mapping total loudness (sone) → `[0, 1)` monotonically.
const LOUDNESS_SATURATION: f32 = 8.0;
/// DIN 45692 sharpness scaling constant (acum).
const SHARPNESS_C: f32 = 0.11;
/// Bark value above which the DIN 45692 sharpness weighting `g(z)` rises above unity.
const SHARPNESS_Z_KNEE: f32 = 16.0;
/// Roughness analysis STFT window length (samples) — short enough to track the temporal
/// envelope modulation that drives roughness.
const ROUGHNESS_WINDOW: usize = 128;
/// Roughness analysis STFT hop (samples); the frame rate `sr/hop` must exceed twice the
/// highest perceptually-relevant modulation frequency (~300 Hz).
const ROUGHNESS_HOP: usize = 32;
/// Modulation frequency of peak roughness sensitivity (Hz, Fastl & Zwicker).
const ROUGHNESS_PEAK_HZ: f32 = 70.0;
/// Half-saturation constant mapping summed roughness → `[0, 1)` monotonically.
const ROUGHNESS_SATURATION: f32 = 2.0;

/// Objective quality metrics system for conversion evaluation
#[derive(Debug, Clone)]
pub struct QualityMetricsSystem {
    /// Reference audio features for comparison
    reference_features: Option<QualityFeatures>,
    /// Perceptual quality model parameters
    perceptual_params: PerceptualParameters,
}

/// Features used for quality assessment
#[derive(Debug, Clone)]
pub struct QualityFeatures {
    /// Spectral features
    pub spectral: Vec<f32>,
    /// Temporal features  
    pub temporal: Vec<f32>,
    /// Prosodic features
    pub prosodic: Vec<f32>,
    /// Perceptual features
    pub perceptual: Vec<f32>,
}

/// Parameters for perceptual quality modeling
#[derive(Debug, Clone)]
pub struct PerceptualParameters {
    /// Weight for spectral similarity
    pub spectral_weight: f32,
    /// Weight for temporal consistency
    pub temporal_weight: f32,
    /// Weight for prosodic preservation
    pub prosodic_weight: f32,
    /// Weight for naturalness
    pub naturalness_weight: f32,
}

impl Default for PerceptualParameters {
    fn default() -> Self {
        Self {
            spectral_weight: 0.3,
            temporal_weight: 0.2,
            prosodic_weight: 0.3,
            naturalness_weight: 0.2,
        }
    }
}

/// Objective quality metrics results
#[derive(Debug, Clone)]
pub struct ObjectiveQualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Spectral similarity score
    pub spectral_similarity: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Prosodic preservation score
    pub prosodic_preservation: f32,
    /// Naturalness score
    pub naturalness: f32,
    /// Perceptual quality score
    pub perceptual_quality: f32,
    /// Signal-to-noise ratio estimate
    pub snr_estimate: f32,
    /// Segmental SNR
    pub segmental_snr: f32,
}

impl QualityMetricsSystem {
    /// Create new quality metrics system
    pub fn new() -> Self {
        Self {
            reference_features: None,
            perceptual_params: PerceptualParameters::default(),
        }
    }

    /// Create with custom perceptual parameters
    pub fn with_perceptual_params(perceptual_params: PerceptualParameters) -> Self {
        Self {
            reference_features: None,
            perceptual_params,
        }
    }

    /// Set reference audio for quality comparison
    pub fn set_reference(&mut self, reference_audio: &[f32], sample_rate: u32) -> Result<()> {
        self.reference_features =
            Some(self.extract_quality_features(reference_audio, sample_rate)?);
        Ok(())
    }

    /// Evaluate objective quality metrics
    pub fn evaluate_quality(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ObjectiveQualityMetrics> {
        debug!(
            "Evaluating objective quality metrics for {} samples",
            audio.len()
        );

        let features = self.extract_quality_features(audio, sample_rate)?;

        let spectral_similarity = if let Some(ref reference) = self.reference_features {
            self.calculate_feature_similarity(&features.spectral, &reference.spectral)
        } else {
            self.estimate_spectral_quality(&features.spectral)
        };

        let temporal_consistency = self.calculate_temporal_consistency(&features.temporal);
        let prosodic_preservation = self.calculate_prosodic_quality(&features.prosodic);
        let naturalness = self.calculate_naturalness(&features.perceptual);
        let perceptual_quality = self.calculate_perceptual_quality(&features);

        let snr_estimate = self.estimate_snr(audio);
        let segmental_snr = self.calculate_segmental_snr(audio);

        // Calculate weighted overall score
        let overall_score = spectral_similarity * self.perceptual_params.spectral_weight
            + temporal_consistency * self.perceptual_params.temporal_weight
            + prosodic_preservation * self.perceptual_params.prosodic_weight
            + naturalness * self.perceptual_params.naturalness_weight;

        info!(
            "Quality evaluation complete: overall_score={:.3}",
            overall_score
        );

        Ok(ObjectiveQualityMetrics {
            overall_score,
            spectral_similarity,
            temporal_consistency,
            prosodic_preservation,
            naturalness,
            perceptual_quality,
            snr_estimate,
            segmental_snr,
        })
    }

    /// Extract quality-relevant features from audio
    fn extract_quality_features(&self, audio: &[f32], sample_rate: u32) -> Result<QualityFeatures> {
        // Extract spectral features (simplified MFCCs)
        let spectral = self.extract_spectral_features(audio);

        // Extract temporal features
        let temporal = self.extract_temporal_features(audio);

        // Extract prosodic features
        let prosodic = self.extract_prosodic_features(audio);

        // Extract perceptual features
        let perceptual = self.extract_perceptual_features(audio, sample_rate);

        Ok(QualityFeatures {
            spectral,
            temporal,
            prosodic,
            perceptual,
        })
    }

    fn extract_spectral_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectral feature extraction
        let mut features = Vec::new();

        let spectrum = self.calculate_power_spectrum(audio);

        // Spectral centroid
        let spectral_centroid = self.calculate_spectral_centroid(&spectrum);
        features.push(spectral_centroid);

        // Spectral rolloff
        let spectral_rolloff = self.calculate_spectral_rolloff(&spectrum);
        features.push(spectral_rolloff);

        // Spectral flatness
        let spectral_flatness = self.calculate_spectral_flatness(&spectrum);
        features.push(spectral_flatness);

        // Add simplified MFCC-like features
        let num_bands = 13;
        let band_energies = self.calculate_mel_band_energies(&spectrum, num_bands);
        features.extend(band_energies);

        features
    }

    fn extract_temporal_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // RMS energy
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        features.push(rms);

        // Zero crossing rate
        let zcr = self.calculate_zero_crossing_rate(audio);
        features.push(zcr);

        // Short-time energy variation
        let energy_variation = self.calculate_energy_variation(audio);
        features.push(energy_variation);

        // Spectral flux
        let spectral_flux = self.calculate_spectral_flux(audio);
        features.push(spectral_flux);

        features
    }

    fn extract_prosodic_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // F0 statistics (simplified)
        let f0_stats = self.calculate_f0_statistics(audio);
        features.extend(f0_stats);

        // Energy contour statistics
        let energy_stats = self.calculate_energy_statistics(audio);
        features.extend(energy_stats);

        // Duration features (simplified)
        let duration_features = self.calculate_duration_features(audio);
        features.extend(duration_features);

        features
    }

    fn extract_perceptual_features(&self, audio: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut features = Vec::new();

        // Loudness estimate (Zwicker / ISO 532-B style)
        let loudness = self.estimate_loudness(audio, sample_rate);
        features.push(loudness);

        // Sharpness estimate (DIN 45692)
        let sharpness = self.estimate_sharpness(audio, sample_rate);
        features.push(sharpness);

        // Roughness estimate (critical-band envelope modulation)
        let roughness = self.estimate_roughness(audio, sample_rate);
        features.push(roughness);

        features
    }

    // Implementation of helper methods for feature extraction

    fn calculate_power_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Hann-windowed FFT magnitude spectrum.
        // n_fft = 512 when enough samples are available; otherwise next power-of-two.
        // Returns magnitude (not squared) for n_fft/2 + 1 bins.
        const N_FFT: usize = POWER_SPECTRUM_N_FFT;
        let empty_len = N_FFT / 2 + 1;

        if audio.len() < 4 {
            return vec![0.0_f32; empty_len];
        }

        let n_fft = if audio.len() >= N_FFT {
            N_FFT
        } else {
            audio.len().next_power_of_two()
        };
        let n_out = n_fft / 2 + 1;

        // Build Hann-windowed, zero-padded f64 buffer for rfft.
        let active = audio.len().min(n_fft);
        let mut buf: Vec<f64> = vec![0.0_f64; n_fft];
        for i in 0..active {
            // Symmetric Hann window: w[i] = 0.5 * (1 - cos(2π·i / (N-1)))
            let w = 0.5
                * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64
                        / (active.saturating_sub(1).max(1)) as f64)
                        .cos());
            buf[i] = audio[i] as f64 * w;
        }

        // Real FFT via scirs2-fft — returns n_fft/2 + 1 complex bins.
        match scirs2_fft::rfft(&buf, Some(n_fft)) {
            Ok(complex_bins) => {
                let mut magnitudes: Vec<f32> = Vec::with_capacity(n_out);
                for c in complex_bins.iter().take(n_out) {
                    magnitudes.push(c.norm() as f32);
                }
                // Pad to the canonical empty_len if n_fft < N_FFT so callers
                // see a consistently-sized spectrum regardless of input length.
                magnitudes.resize(empty_len, 0.0_f32);
                magnitudes
            }
            Err(_) => vec![0.0_f32; empty_len],
        }
    }

    fn calculate_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &energy) in spectrum.iter().enumerate() {
            weighted_sum += (i as f32) * energy;
            total_energy += energy;
        }

        if total_energy > 0.0 {
            weighted_sum / total_energy
        } else {
            0.0
        }
    }

    fn calculate_spectral_rolloff(&self, spectrum: &[f32]) -> f32 {
        let total_energy: f32 = spectrum.iter().sum();
        let threshold = total_energy * 0.85; // 85% rolloff

        let mut cumulative_energy = 0.0;
        for (i, &energy) in spectrum.iter().enumerate() {
            cumulative_energy += energy;
            if cumulative_energy >= threshold {
                return i as f32 / spectrum.len() as f32;
            }
        }

        1.0
    }

    fn calculate_spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.is_empty() {
            return 0.0;
        }

        let geometric_mean = spectrum
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.ln())
            .sum::<f32>()
            / spectrum.len() as f32;

        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    fn calculate_mel_band_energies(&self, spectrum: &[f32], num_bands: usize) -> Vec<f32> {
        let mut band_energies = vec![0.0; num_bands];
        let band_size = spectrum.len() / num_bands;

        #[allow(clippy::needless_range_loop)]
        for band in 0..num_bands {
            let start = band * band_size;
            let end = ((band + 1) * band_size).min(spectrum.len());

            #[allow(clippy::needless_range_loop)]
            for i in start..end {
                band_energies[band] += spectrum[i];
            }

            if end > start {
                band_energies[band] /= (end - start) as f32;
            }
        }

        band_energies
    }

    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }

        let crossings = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();

        crossings as f32 / (audio.len() - 1) as f32
    }

    fn calculate_energy_variation(&self, audio: &[f32]) -> f32 {
        let window_size = audio.len() / 10;
        if window_size < 10 {
            return 0.0;
        }

        let mut energies = Vec::new();
        for i in (0..audio.len()).step_by(window_size) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            energies.push(energy / (end - i) as f32);
        }

        if energies.len() < 2 {
            return 0.0;
        }

        let mean = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance =
            energies.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / energies.len() as f32;

        variance.sqrt()
    }

    fn calculate_spectral_flux(&self, audio: &[f32]) -> f32 {
        let window_size = 256;
        let hop_size = window_size / 2;

        let mut prev_spectrum: Vec<f32> = Vec::new();
        let mut flux_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let spectrum = self.calculate_power_spectrum(window);

            if !prev_spectrum.is_empty() {
                let mut flux = 0.0;
                for (curr, &prev) in spectrum.iter().zip(prev_spectrum.iter()) {
                    flux += (curr - prev).max(0.0);
                }
                flux_values.push(flux);
            }

            prev_spectrum = spectrum;
        }

        if flux_values.is_empty() {
            0.0
        } else {
            flux_values.iter().sum::<f32>() / flux_values.len() as f32
        }
    }

    fn calculate_f0_statistics(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified F0 extraction and statistics
        let window_size = 1024;
        let hop_size = window_size / 2;
        let mut f0_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let f0 = self.estimate_f0_simple(window);
            if f0 > 0.0 {
                f0_values.push(f0);
            }
        }

        if f0_values.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let min_f0 = f0_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f0 = f0_values.iter().fold(0.0f32, |a, &b| a.max(b));

        vec![mean_f0, min_f0, max_f0]
    }

    fn estimate_f0_simple(&self, audio: &[f32]) -> f32 {
        // Normalized autocorrelation-based F0 estimation.
        // Lag bounds are derived from the default sample rate so that the
        // frequency search is always in [F0_MIN_HZ, F0_MAX_HZ].
        let sr = DEFAULT_SAMPLE_RATE;
        let min_lag = (sr / F0_MAX_HZ).floor() as usize; // sr/800 ≈ 27 at 22050
        let max_lag = (sr / F0_MIN_HZ).ceil() as usize; // sr/80  ≈ 276 at 22050

        let n = audio.len();
        if n < min_lag * 2 + 1 {
            return 0.0;
        }

        // Mean-centre the window to remove DC offset.
        let mean = audio.iter().sum::<f32>() / n as f32;
        let x: Vec<f32> = audio.iter().map(|&s| s - mean).collect();

        // Energy of the entire centred window — used as normalisation anchor.
        let energy_full: f32 = x.iter().map(|v| v * v).sum();
        if energy_full < f32::EPSILON {
            return 0.0; // Silence
        }

        let effective_max_lag = max_lag.min(n / 2);
        if effective_max_lag <= min_lag {
            return 0.0;
        }

        let mut best_lag = 0usize;
        let mut best_r = f32::NEG_INFINITY;

        for tau in min_lag..=effective_max_lag {
            // Number of overlapping samples for this lag.
            let overlap = n - tau;
            // Unnormalised cross-correlation at lag τ.
            let cross: f32 = (0..overlap).map(|i| x[i] * x[i + tau]).sum();
            // Energy of the lagged sub-window (denominator normalisation).
            let energy_lag: f32 = (tau..n).map(|i| x[i] * x[i]).sum();
            // Energy of the un-lagged sub-window.
            let energy_base: f32 = (0..overlap).map(|i| x[i] * x[i]).sum();

            let denom = (energy_base * energy_lag).sqrt();
            let r = if denom > f32::EPSILON {
                cross / denom
            } else {
                0.0
            };

            if r > best_r {
                best_r = r;
                best_lag = tau;
            }
        }

        if best_r > VOICED_THRESHOLD && best_lag > 0 {
            sr / best_lag as f32
        } else {
            0.0
        }
    }

    fn calculate_energy_statistics(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 256;
        let mut energies = Vec::new();

        for i in (0..audio.len()).step_by(window_size / 2) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            energies.push(energy / (end - i) as f32);
        }

        if energies.is_empty() {
            return vec![0.0, 0.0];
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let energy_variance = energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        vec![mean_energy, energy_variance.sqrt()]
    }

    fn calculate_duration_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified duration features
        let total_duration = audio.len() as f32;
        let non_silent_samples = audio.iter().filter(|&&x| x.abs() > 0.01).count() as f32;

        let speech_rate = if total_duration > 0.0 {
            non_silent_samples / total_duration
        } else {
            0.0
        };

        vec![speech_rate]
    }

    /// Estimate perceived loudness with a Zwicker / ISO 532-B style critical-band model.
    ///
    /// The magnitude spectrum (reused from `calculate_power_spectrum`) is squared into
    /// power, mapped onto the Bark critical-band scale, spread with Schroeder's spreading
    /// function, and converted to specific loudness `N'(z)` via Stevens' compressive power
    /// law `N'(z) ∝ (E/E0)^0.23`. The total loudness `N = Σ N'(z)·ΔBark` (sone) is mapped
    /// through the monotonic soft-saturation `N / (N + K)` so the result stays in `[0, 1)`
    /// while remaining a strictly increasing function of true loudness — not raw RMS.
    fn estimate_loudness(&self, audio: &[f32], sample_rate: u32) -> f32 {
        let (_specific, total_loudness) = self.specific_loudness(audio, sample_rate);
        total_loudness / (total_loudness + LOUDNESS_SATURATION)
    }

    /// Estimate sharpness (acum) following DIN 45692.
    ///
    /// `S = c · Σ(N'(z)·g(z)·z·ΔBark) / Σ(N'(z)·ΔBark)`, where the band-weighting `g(z)` is
    /// unity up to `SHARPNESS_Z_KNEE` Bark and rises as `0.066·exp(0.171·z)` above it, and
    /// `c = SHARPNESS_C`. The specific loudness `N'(z)` is shared with the loudness model so
    /// high-frequency energy raises sharpness through the Bark-weighted loudness centroid.
    fn estimate_sharpness(&self, audio: &[f32], sample_rate: u32) -> f32 {
        let (specific, _total) = self.specific_loudness(audio, sample_rate);

        let mut numerator = 0.0_f32;
        let mut denominator = 0.0_f32;
        for (band, &n_prime) in specific.iter().enumerate() {
            // Band centre on the Bark scale.
            let z = (band as f32 + 0.5) * BARK_BAND_WIDTH;
            let g = if z <= SHARPNESS_Z_KNEE {
                1.0
            } else {
                0.066 * (0.171 * z).exp()
            };
            numerator += n_prime * g * z * BARK_BAND_WIDTH;
            denominator += n_prime * BARK_BAND_WIDTH;
        }

        if denominator > f32::EPSILON {
            SHARPNESS_C * numerator / denominator
        } else {
            0.0
        }
    }

    /// Estimate roughness from critical-band temporal-envelope modulation.
    ///
    /// A short-window STFT (`ROUGHNESS_WINDOW` / `ROUGHNESS_HOP`) yields a per-frame energy
    /// time series — the temporal envelope — for each Bark band. The modulation spectrum of
    /// every band envelope is taken with `scirs2_fft::rfft`; each AC component is normalised
    /// by the envelope DC to form a modulation-depth index, squared, and weighted by the
    /// roughness modulation-frequency curve `roughness_weight` (peaking near
    /// `ROUGHNESS_PEAK_HZ`). The summed contribution is soft-saturated into a stable `[0, 1)`
    /// scalar. A Hann window is applied to each envelope before the FFT to suppress leakage;
    /// its coherent gain cancels in the AC/DC ratio, so modulation depth is preserved.
    fn estimate_roughness(&self, audio: &[f32], sample_rate: u32) -> f32 {
        let envelopes = self.band_energy_envelopes(audio, sample_rate);
        let n_frames = envelopes.first().map_or(0, Vec::len);
        if n_frames < 4 {
            return 0.0;
        }

        let frame_rate = sample_rate as f32 / ROUGHNESS_HOP as f32;
        let n_pad = n_frames.next_power_of_two();

        // Hann window over the envelope time series (length is uniform across bands).
        let hann: Vec<f64> = (0..n_frames)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64 / (n_frames - 1).max(1) as f64).cos())
            })
            .collect();

        let mut total_roughness = 0.0_f32;
        for envelope in &envelopes {
            // Hann-window the envelope, then zero-pad (tapered ends keep padding artefact-free).
            let mut buf: Vec<f64> = envelope
                .iter()
                .zip(hann.iter())
                .map(|(&e, &w)| e as f64 * w)
                .collect();
            buf.resize(n_pad, 0.0_f64);

            let spectrum = match scirs2_fft::rfft(&buf, Some(n_pad)) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // DC term anchors the modulation-depth normalisation.
            let dc = spectrum.first().map_or(0.0_f32, |c| c.norm() as f32);
            if dc <= f32::EPSILON {
                continue;
            }

            let mut band_roughness = 0.0_f32;
            for (k, c) in spectrum.iter().enumerate().skip(1) {
                let f_mod = k as f32 * frame_rate / n_pad as f32;
                let depth = c.norm() as f32 / dc;
                band_roughness += depth * depth * roughness_weight(f_mod);
            }
            total_roughness += band_roughness;
        }

        total_roughness / (total_roughness + ROUGHNESS_SATURATION)
    }

    /// Spread per-band excitation (linear power) on the Bark critical-band scale.
    ///
    /// FFT magnitudes from `calculate_power_spectrum` are squared into power, accumulated
    /// into `NUM_BARK_BANDS` critical bands via the Zwicker Bark mapping, then convolved
    /// with Schroeder's spreading function to model upward/downward auditory masking.
    fn bark_band_excitation(&self, audio: &[f32], sample_rate: u32) -> Vec<f32> {
        let magnitudes = self.calculate_power_spectrum(audio);
        let sr = sample_rate as f32;

        // Reconstruct the FFT size used by `calculate_power_spectrum` so bin → frequency
        // mapping is exact. Zero-padded bins (short inputs) carry no power and stay harmless.
        let n_fft = if audio.len() >= POWER_SPECTRUM_N_FFT {
            POWER_SPECTRUM_N_FFT
        } else {
            audio.len().next_power_of_two().max(2)
        };

        let mut excitation = vec![0.0_f32; NUM_BARK_BANDS];
        for (i, &mag) in magnitudes.iter().enumerate() {
            let freq = i as f32 * sr / n_fft as f32;
            let band = (hz_to_bark(freq) / BARK_BAND_WIDTH) as usize;
            if band < NUM_BARK_BANDS {
                // power = magnitude²
                excitation[band] += mag * mag;
            }
        }

        // Convolve with Schroeder's spreading function (linear power domain).
        let mut spread = vec![0.0_f32; NUM_BARK_BANDS];
        for (target, slot) in spread.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (source, &e) in excitation.iter().enumerate() {
                let delta_z = (target as f32 - source as f32) * BARK_BAND_WIDTH;
                acc += e * spreading_function(delta_z);
            }
            *slot = acc;
        }
        spread
    }

    /// Specific loudness `N'(z)` per Bark band (sone/Bark) and total loudness `N` (sone).
    ///
    /// Stevens' compressive power law with a threshold-in-quiet floor:
    /// `N'(z) = max(0, (E/E0)^0.23 - (E_TQ/E0)^0.23)`.
    fn specific_loudness(&self, audio: &[f32], sample_rate: u32) -> (Vec<f32>, f32) {
        let excitation = self.bark_band_excitation(audio, sample_rate);
        let threshold_term = (LOUDNESS_E_THRESHOLD / LOUDNESS_E0).powf(LOUDNESS_EXPONENT);

        let specific: Vec<f32> = excitation
            .iter()
            .map(|&e| ((e / LOUDNESS_E0).powf(LOUDNESS_EXPONENT) - threshold_term).max(0.0))
            .collect();
        let total = specific.iter().sum::<f32>() * BARK_BAND_WIDTH;
        (specific, total)
    }

    /// Per-Bark-band temporal-envelope time series obtained from a short-window STFT.
    ///
    /// Returns one envelope (per-frame band energy) for each of `NUM_BARK_BANDS` critical
    /// bands; all envelopes share the same length (one sample per analysis frame). The short
    /// window and hop give a frame rate high enough to resolve amplitude modulation across
    /// the roughness-relevant range (up to ~300 Hz).
    fn band_energy_envelopes(&self, audio: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
        let window = ROUGHNESS_WINDOW;
        let hop = ROUGHNESS_HOP;
        if audio.len() < window {
            return Vec::new();
        }

        let sr = sample_rate as f32;
        let n_out = window / 2 + 1;
        let mut envelopes: Vec<Vec<f32>> = vec![Vec::new(); NUM_BARK_BANDS];

        // Per-frame Hann window.
        let hann: Vec<f64> = (0..window)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (window - 1) as f64).cos())
            })
            .collect();

        // Precompute bin → Bark band mapping for the analysis window.
        let bin_band: Vec<Option<usize>> = (0..n_out)
            .map(|i| {
                let freq = i as f32 * sr / window as f32;
                let band = (hz_to_bark(freq) / BARK_BAND_WIDTH) as usize;
                (band < NUM_BARK_BANDS).then_some(band)
            })
            .collect();

        let mut start = 0;
        while start + window <= audio.len() {
            let buf: Vec<f64> = (0..window)
                .map(|i| audio[start + i] as f64 * hann[i])
                .collect();
            if let Ok(spectrum) = scirs2_fft::rfft(&buf, Some(window)) {
                let mut band_energy = vec![0.0_f32; NUM_BARK_BANDS];
                for (c, &band_opt) in spectrum.iter().take(n_out).zip(bin_band.iter()) {
                    if let Some(band) = band_opt {
                        let mag = c.norm() as f32;
                        band_energy[band] += mag * mag;
                    }
                }
                for (band, e) in band_energy.into_iter().enumerate() {
                    envelopes[band].push(e);
                }
            }
            start += hop;
        }

        envelopes
    }

    // Quality calculation methods

    fn calculate_feature_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut similarity = 0.0;
        for i in 0..min_len {
            let diff = (features1[i] - features2[i]).abs();
            similarity += 1.0 / (1.0 + diff);
        }

        similarity / min_len as f32
    }

    fn estimate_spectral_quality(&self, features: &[f32]) -> f32 {
        // Estimate quality without reference based on feature characteristics
        if features.is_empty() {
            return 0.0;
        }

        // Check for typical speech-like spectral characteristics
        let centroid = features[0];
        let rolloff = if features.len() > 1 { features[1] } else { 0.5 };

        // Ideal values for speech
        let centroid_quality =
            1.0 - ((centroid / features.len() as f32 - 0.3).abs() / 0.3).min(1.0);
        let rolloff_quality = 1.0 - ((rolloff - 0.8).abs() / 0.2).min(1.0);

        (centroid_quality + rolloff_quality) / 2.0
    }

    fn calculate_temporal_consistency(&self, temporal_features: &[f32]) -> f32 {
        if temporal_features.len() < 3 {
            return 1.0;
        }

        // Check energy variation (less variation = better consistency)
        let energy_variation = temporal_features[2];
        1.0 - energy_variation.min(1.0)
    }

    fn calculate_prosodic_quality(&self, prosodic_features: &[f32]) -> f32 {
        if prosodic_features.len() < 3 {
            return 0.5;
        }

        let mean_f0 = prosodic_features[0];
        let f0_range = prosodic_features[2] - prosodic_features[1];

        // Check if F0 is in typical speech range
        let f0_quality = if mean_f0 >= 80.0 && mean_f0 <= 400.0 {
            1.0 - ((mean_f0 - 150.0).abs() / 150.0).min(1.0)
        } else {
            0.0
        };

        // Check if F0 range is reasonable
        let range_quality = if f0_range >= 10.0 && f0_range <= 100.0 {
            1.0 - ((f0_range - 30.0).abs() / 30.0).min(1.0)
        } else {
            0.0
        };

        (f0_quality + range_quality) / 2.0
    }

    fn calculate_naturalness(&self, perceptual_features: &[f32]) -> f32 {
        if perceptual_features.is_empty() {
            return 0.5;
        }

        let loudness = perceptual_features[0];
        let roughness = if perceptual_features.len() > 2 {
            perceptual_features[2]
        } else {
            0.5
        };

        // Natural speech should have moderate loudness and low roughness
        let loudness_quality = 1.0 - ((loudness - 0.3).abs() / 0.3).min(1.0);
        let roughness_quality = 1.0 - roughness;

        (loudness_quality + roughness_quality) / 2.0
    }

    fn calculate_perceptual_quality(&self, features: &QualityFeatures) -> f32 {
        // Combine all feature types for overall perceptual quality
        let spectral_quality = self.estimate_spectral_quality(&features.spectral);
        let temporal_quality = self.calculate_temporal_consistency(&features.temporal);
        let prosodic_quality = self.calculate_prosodic_quality(&features.prosodic);
        let naturalness = self.calculate_naturalness(&features.perceptual);

        // Weighted combination
        spectral_quality * 0.3 + temporal_quality * 0.2 + prosodic_quality * 0.3 + naturalness * 0.2
    }

    fn estimate_snr(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Estimate signal power (simplified)
        let signal_power: f32 = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;

        // Estimate noise power from quiet segments (simplified)
        let sorted_samples: Vec<f32> = {
            let mut samples = audio.iter().map(|&x| x.abs()).collect::<Vec<f32>>();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            samples
        };

        let noise_threshold_idx = sorted_samples.len() / 4; // Bottom quartile as noise
        let noise_power = if noise_threshold_idx < sorted_samples.len() {
            sorted_samples[noise_threshold_idx] * sorted_samples[noise_threshold_idx]
        } else {
            0.001 // Small noise floor
        };

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // High SNR when no noise detected
        }
    }

    fn calculate_segmental_snr(&self, audio: &[f32]) -> f32 {
        let segment_size = 256;
        let mut snr_values = Vec::new();

        for i in (0..audio.len()).step_by(segment_size) {
            let end = (i + segment_size).min(audio.len());
            let segment = &audio[i..end];

            if !segment.is_empty() {
                let segment_snr = self.estimate_snr(segment);
                snr_values.push(segment_snr);
            }
        }

        if snr_values.is_empty() {
            0.0
        } else {
            snr_values.iter().sum::<f32>() / snr_values.len() as f32
        }
    }
}

impl Default for QualityMetricsSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Psychoacoustic helper functions (Bark scale, masking, roughness weighting)
// ---------------------------------------------------------------------------

/// Convert a frequency (Hz) to the Bark critical-band scale (Zwicker & Terhardt, 1980).
///
/// `z = 13·atan(0.00076·f) + 3.5·atan((f/7500)²)` [Bark].
fn hz_to_bark(freq_hz: f32) -> f32 {
    let f = freq_hz.max(0.0);
    13.0 * (0.000_76 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

/// Schroeder's (1979) auditory spreading function, returned as a linear power gain.
///
/// `10·log10(SF) = 15.81 + 7.5·(Δz + 0.474) - 17.5·sqrt(1 + (Δz + 0.474)²)` [dB]. Models the
/// asymmetric upward/downward spread of masking between adjacent critical bands.
fn spreading_function(delta_z: f32) -> f32 {
    let x = delta_z + 0.474;
    let db = 15.81 + 7.5 * x - 17.5 * (1.0 + x * x).sqrt();
    10.0_f32.powf(db / 10.0)
}

/// Roughness sensitivity as a function of envelope modulation frequency (Hz).
///
/// A gamma-shaped band-pass weighting that is zero at DC, unity at `ROUGHNESS_PEAK_HZ`
/// (Fastl & Zwicker locate maximum roughness near 70 Hz modulation), and decays for higher
/// modulation frequencies: `w(f) = (f/f0)·exp(1 - f/f0)`.
fn roughness_weight(f_mod: f32) -> f32 {
    if f_mod <= 0.0 {
        return 0.0;
    }
    let u = f_mod / ROUGHNESS_PEAK_HZ;
    u * (1.0 - u).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq_hz: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Power-spectrum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_power_spectrum_has_correct_length() {
        let sys = QualityMetricsSystem::new();
        // 512 samples → n_fft = 512 → n_out = 512/2 + 1 = 257
        let audio: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
        let spec = sys.calculate_power_spectrum(&audio);
        assert_eq!(spec.len(), 257, "expected 257 bins for 512-sample input");
    }

    #[test]
    fn test_power_spectrum_single_tone_peak() {
        // 440 Hz sine at 22050 Hz sample rate.
        // Expected peak bin ≈ 440 * 512 / 22050 ≈ 10.2 → bin 10.
        let sys = QualityMetricsSystem::new();
        let sr = 22050.0_f32;
        let audio = make_sine(440.0, sr, 512);
        let spec = sys.calculate_power_spectrum(&audio);

        let peak_bin = spec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Allow ±2 bins of tolerance for Hann windowing leakage.
        let expected_bin = (440.0 * 512.0 / sr).round() as usize;
        assert!(
            peak_bin.abs_diff(expected_bin) <= 2,
            "peak bin {peak_bin} is not close to expected bin {expected_bin}"
        );
    }

    // -----------------------------------------------------------------------
    // F0 estimation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f0_estimation_voiced() {
        // 220 Hz sine at 22050 Hz sample rate — well within [F0_MIN_HZ, F0_MAX_HZ].
        let sys = QualityMetricsSystem::new();
        let audio = make_sine(220.0, DEFAULT_SAMPLE_RATE, 4096);
        let f0 = sys.estimate_f0_simple(&audio);
        assert!(
            f0 >= 180.0 && f0 <= 260.0,
            "expected F0 between 180 and 260 Hz, got {f0}"
        );
    }

    #[test]
    fn test_f0_estimation_silence() {
        let sys = QualityMetricsSystem::new();
        let silence = vec![0.0_f32; 4096];
        let f0 = sys.estimate_f0_simple(&silence);
        assert_eq!(f0, 0.0, "silence should return F0=0.0");
    }

    // -----------------------------------------------------------------------
    // Psychoacoustic (Bark critical-band) tests
    // -----------------------------------------------------------------------

    /// Amplitude-modulated tone: carrier `carrier_hz` modulated at `mod_hz` with depth.
    fn make_am_sine(
        carrier_hz: f32,
        mod_hz: f32,
        depth: f32,
        sample_rate: f32,
        num_samples: usize,
    ) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let envelope = 1.0 + depth * (2.0 * std::f32::consts::PI * mod_hz * t).sin();
                envelope * (2.0 * std::f32::consts::PI * carrier_hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_sharpness_high_tone_sharper_than_low_tone() {
        // DIN 45692: a 1 kHz tone sits at a higher Bark centroid than a 200 Hz tone,
        // so it must be measured as sharper.
        let sys = QualityMetricsSystem::new();
        let sr: u32 = 22050;
        let high = make_sine(1000.0, sr as f32, 8192);
        let low = make_sine(200.0, sr as f32, 8192);

        let sharp_high = sys.estimate_sharpness(&high, sr);
        let sharp_low = sys.estimate_sharpness(&low, sr);

        assert!(
            sharp_high > sharp_low,
            "1 kHz tone (S={sharp_high}) should be sharper than 200 Hz tone (S={sharp_low})"
        );
    }

    #[test]
    fn test_roughness_modulated_rougher_than_unmodulated() {
        // A 70 Hz amplitude-modulated tone sits at the roughness sensitivity peak and must
        // be rougher than the same carrier with no modulation.
        let sys = QualityMetricsSystem::new();
        let sr: u32 = 22050;
        let modulated = make_am_sine(1000.0, 70.0, 0.9, sr as f32, 16384);
        let plain = make_sine(1000.0, sr as f32, 16384);

        let rough_mod = sys.estimate_roughness(&modulated, sr);
        let rough_plain = sys.estimate_roughness(&plain, sr);

        assert!(
            rough_mod > rough_plain,
            "70 Hz AM tone (R={rough_mod}) should be rougher than unmodulated tone (R={rough_plain})"
        );
    }

    #[test]
    fn test_loudness_monotonic_in_amplitude() {
        // Loudness must increase with level — a property RMS shares but which here must
        // survive the full Bark-band / power-law / saturation pipeline.
        let sys = QualityMetricsSystem::new();
        let sr: u32 = 22050;
        let base = make_sine(440.0, sr as f32, 8192);
        let quiet: Vec<f32> = base.iter().map(|s| s * 0.1).collect();
        let loud: Vec<f32> = base.iter().map(|s| s * 0.5).collect();

        let loud_loudness = sys.estimate_loudness(&loud, sr);
        let quiet_loudness = sys.estimate_loudness(&quiet, sr);

        assert!(
            loud_loudness > quiet_loudness,
            "louder input (L={loud_loudness}) should exceed quieter input (L={quiet_loudness})"
        );
    }

    #[test]
    fn test_psychoacoustic_outputs_finite_and_nonnegative() {
        let sys = QualityMetricsSystem::new();
        let sr: u32 = 22050;
        let audio = make_am_sine(800.0, 50.0, 0.7, sr as f32, 8192);

        let loudness = sys.estimate_loudness(&audio, sr);
        let sharpness = sys.estimate_sharpness(&audio, sr);
        let roughness = sys.estimate_roughness(&audio, sr);

        for (name, value) in [
            ("loudness", loudness),
            ("sharpness", sharpness),
            ("roughness", roughness),
        ] {
            assert!(value.is_finite(), "{name} must be finite, got {value}");
            assert!(value >= 0.0, "{name} must be non-negative, got {value}");
        }

        // Silence must be well-defined and yield zero loudness.
        let silence = vec![0.0_f32; 8192];
        assert_eq!(sys.estimate_loudness(&silence, sr), 0.0);
        assert!(sys.estimate_roughness(&silence, sr).is_finite());
    }
}
