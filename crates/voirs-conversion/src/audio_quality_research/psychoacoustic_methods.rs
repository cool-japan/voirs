//! Psychoacoustic analysis implementation methods

use super::researcher::AudioQualityResearcher;
use super::types::{ResearchCriticalBandAnalysis, TonalityAnalysis};
use crate::Error;

impl AudioQualityResearcher {
    pub(crate) fn calculate_loudness_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_loudness =
            (original.iter().map(|&x| x * x).sum::<f32>() / original.len() as f32).sqrt();
        let proc_loudness =
            (processed.iter().map(|&x| x * x).sum::<f32>() / processed.len() as f32).sqrt();
        Ok((orig_loudness - proc_loudness).abs())
    }

    /// Analyze the 24 Bark critical bands of `original` vs `processed`.
    ///
    /// Both signals' magnitude spectra are partitioned into 24 contiguous bands
    /// (the same partitioning used by [`Self::calculate_masking_threshold_deviation`]).
    /// For each band the per-band deviation is the relative energy change
    /// `|orig_energy - proc_energy| / orig_energy`. The `overall_distortion` is the
    /// mean of those deviations, while `hf_preservation` / `lf_preservation` are the
    /// processed/original energy ratios summed over the upper / lower halves of the
    /// band range. Identical inputs therefore produce ~0 distortion and preservation
    /// ratios ~1, and spectrally different inputs produce non-trivial per-band values.
    pub(crate) fn analyze_critical_bands(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<ResearchCriticalBandAnalysis, Error> {
        const NUM_BANDS: usize = 24; // 24 Bark bands

        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        // With an empty/degenerate spectrum, report no distortion and full preservation.
        if orig_spectrum.is_empty() {
            return Ok(ResearchCriticalBandAnalysis {
                band_deviations: vec![0.0; NUM_BANDS],
                overall_distortion: 0.0,
                hf_preservation: 1.0,
                lf_preservation: 1.0,
            });
        }

        let num_bands = orig_spectrum.len().min(NUM_BANDS).max(1);
        let band_size = (orig_spectrum.len() / num_bands).max(1);

        let mut band_deviations = vec![0.0_f32; NUM_BANDS];
        let mut orig_band_energies = vec![0.0_f32; num_bands];
        let mut proc_band_energies = vec![0.0_f32; num_bands];

        for band in 0..num_bands {
            let start_bin = band * band_size;
            let end_bin = if band == num_bands - 1 {
                orig_spectrum.len()
            } else {
                ((band + 1) * band_size).min(orig_spectrum.len())
            };

            if start_bin >= end_bin {
                continue;
            }

            let orig_energy: f32 = orig_spectrum[start_bin..end_bin]
                .iter()
                .map(|&x| x * x)
                .sum();
            let proc_energy: f32 = proc_spectrum
                .get(start_bin..end_bin.min(proc_spectrum.len()))
                .map(|s| s.iter().map(|&x| x * x).sum())
                .unwrap_or(0.0);

            orig_band_energies[band] = orig_energy;
            proc_band_energies[band] = proc_energy;

            // Relative per-band deviation. When the original band is essentially
            // silent the deviation is defined as the processed energy presence.
            band_deviations[band] = if orig_energy > 1e-10 {
                ((orig_energy - proc_energy).abs() / orig_energy).clamp(0.0, 2.0)
            } else if proc_energy > 1e-10 {
                1.0
            } else {
                0.0
            };
        }

        let overall_distortion =
            band_deviations[..num_bands].iter().sum::<f32>() / num_bands as f32;

        // Split the band range into lower and upper halves for LF / HF preservation.
        let half = num_bands / 2;
        let lf_orig: f32 = orig_band_energies[..half.max(1)].iter().sum();
        let lf_proc: f32 = proc_band_energies[..half.max(1)].iter().sum();
        let hf_orig: f32 = orig_band_energies[half..].iter().sum();
        let hf_proc: f32 = proc_band_energies[half..].iter().sum();

        let lf_preservation = if lf_orig > 1e-10 {
            (lf_proc / lf_orig).clamp(0.0, 2.0)
        } else {
            1.0
        };
        let hf_preservation = if hf_orig > 1e-10 {
            (hf_proc / hf_orig).clamp(0.0, 2.0)
        } else {
            1.0
        };

        Ok(ResearchCriticalBandAnalysis {
            band_deviations,
            overall_distortion,
            hf_preservation,
            lf_preservation,
        })
    }

    pub(crate) fn calculate_masking_threshold_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        if original.len() != processed.len() || original.is_empty() {
            return Ok(0.0);
        }

        // Calculate critical band energies for both signals
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let num_bands = orig_spectrum.len().min(24); // Use up to 24 critical bands
        let band_size = orig_spectrum.len() / num_bands;

        let mut total_deviation = 0.0;
        let mut valid_bands = 0;

        for band in 0..num_bands {
            let start_bin = band * band_size;
            let end_bin = ((band + 1) * band_size).min(orig_spectrum.len());

            if start_bin >= end_bin {
                continue;
            }

            // Calculate band energy
            let orig_energy: f32 = orig_spectrum[start_bin..end_bin]
                .iter()
                .map(|&x| x * x)
                .sum();
            let proc_energy: f32 = proc_spectrum[start_bin..end_bin]
                .iter()
                .map(|&x| x * x)
                .sum();

            if orig_energy <= 1e-10 {
                continue;
            }

            // Calculate masking threshold for this band
            let mut masking_threshold = 0.0;

            // Simultaneous masking from neighboring bands
            for other_band in 0..num_bands {
                if other_band == band {
                    continue;
                }

                let other_start = other_band * band_size;
                let other_end = ((other_band + 1) * band_size).min(orig_spectrum.len());
                let other_energy: f32 = orig_spectrum[other_start..other_end]
                    .iter()
                    .map(|&x| x * x)
                    .sum();

                if other_energy > 1e-10 {
                    let distance = (band as i32 - other_band as i32).abs() as f32;
                    let spreading = (-0.15 * distance).exp(); // Masking spread function
                    masking_threshold += other_energy * spreading;
                }
            }

            // Calculate threshold deviation
            let orig_threshold = orig_energy * 0.1 + masking_threshold;
            let proc_threshold = proc_energy * 0.1 + masking_threshold;

            if orig_threshold > 1e-10 {
                let deviation = ((proc_threshold - orig_threshold) / orig_threshold).abs();
                total_deviation += deviation;
                valid_bands += 1;
            }
        }

        if valid_bands > 0 {
            Ok((total_deviation / valid_bands as f32).clamp(0.0, 2.0))
        } else {
            Ok(0.0)
        }
    }

    pub(crate) fn calculate_sharpness_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        if original.len() != processed.len() || original.is_empty() {
            return Ok(0.0);
        }

        let orig_sharpness = self.calculate_sharpness(original)?;
        let proc_sharpness = self.calculate_sharpness(processed)?;

        let max_sharpness = orig_sharpness.max(proc_sharpness);
        if max_sharpness > 1e-10 {
            Ok(((orig_sharpness - proc_sharpness) / max_sharpness).abs())
        } else {
            Ok(0.0)
        }
    }

    pub(crate) fn calculate_roughness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let mut roughness = 0.0;
        let modulation_freq_min = 15.0; // Hz
        let modulation_freq_max = 300.0; // Hz

        // Calculate roughness based on amplitude modulation in critical bands
        let num_bands = spectrum.len().min(24);
        let band_size = spectrum.len() / num_bands;

        for band in 0..num_bands {
            let start_bin = band * band_size;
            let end_bin = ((band + 1) * band_size).min(spectrum.len());

            if start_bin >= end_bin {
                continue;
            }

            let band_energy: f32 = spectrum[start_bin..end_bin].iter().map(|&x| x * x).sum();

            if band_energy <= 1e-10 {
                continue;
            }

            // Simulate amplitude modulation detection
            let band_center_freq = (start_bin + end_bin) as f32 / 2.0;

            // Look for fluctuations in the roughness-sensitive frequency range
            for mod_freq in [20.0, 40.0, 70.0, 150.0, 250.0] {
                if mod_freq >= modulation_freq_min && mod_freq <= modulation_freq_max {
                    // Roughness function approximation (simplified Zwicker model)
                    let roughness_contribution = band_energy
                        * (mod_freq / 70.0f32).powf(-0.8)
                        * (-0.3 * (band_center_freq / 1000.0)).exp();
                    roughness += roughness_contribution;
                }
            }
        }

        // Normalize roughness to 0-1 range
        Ok((roughness * 0.1).clamp(0.0, 1.0))
    }

    pub(crate) fn calculate_fluctuation_strength(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Calculate fluctuation strength based on low-frequency amplitude modulation
        let envelope = self.calculate_envelope(audio);
        if envelope.len() < 10 {
            return Ok(0.0);
        }

        // Calculate modulation spectrum of the envelope
        let mut fluctuation_strength = 0.0;
        let target_mod_freq = 4.0; // Hz - maximum fluctuation strength

        // Simple envelope analysis for fluctuation detection
        let mut envelope_variations = Vec::new();
        let window_size = envelope.len() / 10;

        if window_size > 0 {
            for i in 0..envelope.len().saturating_sub(window_size) {
                let current_window = &envelope[i..i + window_size];
                let mean_energy = current_window.iter().sum::<f32>() / window_size as f32;
                let variance = current_window
                    .iter()
                    .map(|&x| (x - mean_energy).powi(2))
                    .sum::<f32>()
                    / window_size as f32;
                envelope_variations.push(variance.sqrt());
            }
        }

        if !envelope_variations.is_empty() {
            // Calculate fluctuation based on envelope variation patterns
            let mean_variation =
                envelope_variations.iter().sum::<f32>() / envelope_variations.len() as f32;

            // Look for periodic patterns in envelope variations (simplified)
            let mut periodic_strength = 0.0;
            for i in 1..envelope_variations.len() {
                let current_var = envelope_variations[i];
                if i > 2 {
                    let prev_var = envelope_variations[i - 2];
                    // Look for repeating patterns
                    if (current_var - prev_var).abs() < mean_variation * 0.5 {
                        periodic_strength += current_var;
                    }
                }
            }

            fluctuation_strength = if mean_variation > 1e-10 {
                (periodic_strength / (envelope_variations.len() as f32 * mean_variation))
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        Ok(fluctuation_strength)
    }

    /// Analyze the tonal vs. noise character of `original` and how well it is
    /// preserved in `processed`.
    ///
    /// The tonality of a signal is derived from its *spectral flatness measure*
    /// (SFM = geometric_mean / arithmetic_mean of the power spectrum). A pure tone
    /// has a near-zero SFM (one dominant bin) → tonality `1 - SFM ≈ 1`, whereas
    /// white noise has SFM near 1 → tonality ≈ 0.
    ///
    /// - `tonal_noise_ratio`: tonality of the original signal, `1 - SFM_orig`.
    /// - `tonal_preservation`: `1 - |tonality_orig - tonality_proc|`, i.e. how
    ///   closely the processed signal retains the original's tonal/noise balance.
    /// - `noise_deviation`: the complementary `|tonality_orig - tonality_proc|`.
    /// - `spectral_peaks_preservation`: correlation between the original and
    ///   processed spectral-peak envelopes (local peaks via [`Self::find_local_peak`]).
    pub(crate) fn analyze_tonality(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<TonalityAnalysis, Error> {
        let orig_tonality = self.spectral_tonality(original);
        let proc_tonality = self.spectral_tonality(processed);

        let noise_deviation = (orig_tonality - proc_tonality).abs().clamp(0.0, 1.0);
        let tonal_preservation = (1.0 - noise_deviation).clamp(0.0, 1.0);

        let spectral_peaks_preservation = self.spectral_peak_preservation(original, processed);

        Ok(TonalityAnalysis {
            tonal_noise_ratio: orig_tonality,
            tonal_preservation,
            noise_deviation,
            spectral_peaks_preservation,
        })
    }

    /// Tonality of a signal in `[0, 1]`, computed as `1 - spectral_flatness` of the
    /// power spectrum. Pure tones → ~1, white noise → ~0.
    fn spectral_tonality(&self, audio: &[f32]) -> f32 {
        let spectrum = self.magnitude_spectrum(audio);
        if spectrum.is_empty() {
            return 0.0;
        }

        // Power spectrum.
        let powers: Vec<f32> = spectrum.iter().map(|&x| x * x).collect();
        let n = powers.len() as f32;

        // Geometric mean via the log-domain mean; arithmetic mean directly.
        // A small floor avoids log(0) and divide-by-zero on silent bins.
        let mut log_sum = 0.0_f32;
        let mut arith_sum = 0.0_f32;
        for &p in &powers {
            let pf = p.max(1e-12);
            log_sum += pf.ln();
            arith_sum += pf;
        }

        let geometric_mean = (log_sum / n).exp();
        let arithmetic_mean = arith_sum / n;

        let flatness = if arithmetic_mean > 1e-12 {
            (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
        } else {
            1.0
        };

        (1.0 - flatness).clamp(0.0, 1.0)
    }

    /// Preservation of spectral peak structure between two signals, in `[0, 1]`.
    ///
    /// Builds a per-bin local-peak envelope of each magnitude spectrum (using
    /// [`Self::find_local_peak`]) and returns their Pearson correlation mapped to
    /// `[0, 1]`. Identical inputs → 1.0.
    fn spectral_peak_preservation(&self, original: &[f32], processed: &[f32]) -> f32 {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let len = orig_spectrum.len().min(proc_spectrum.len());
        if len == 0 {
            return 1.0;
        }

        const RADIUS: usize = 2;
        let orig_peaks: Vec<f32> = (0..len)
            .map(|bin| self.find_local_peak(&orig_spectrum, bin, RADIUS))
            .collect();
        let proc_peaks: Vec<f32> = (0..len)
            .map(|bin| self.find_local_peak(&proc_spectrum, bin, RADIUS))
            .collect();

        // Map correlation from [-1, 1] to [0, 1]; constant envelopes correlate at 1.0.
        let correlation = self.calculate_correlation(&orig_peaks, &proc_peaks);
        ((correlation + 1.0) * 0.5).clamp(0.0, 1.0)
    }

    pub(crate) fn calculate_loudness_difference(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_rms =
            (original.iter().map(|&x| x * x).sum::<f32>() / original.len() as f32).sqrt();
        let proc_rms =
            (processed.iter().map(|&x| x * x).sum::<f32>() / processed.len() as f32).sqrt();
        Ok((orig_rms - proc_rms).abs() / orig_rms.max(1e-10))
    }

    /// Absolute difference in psychoacoustic sharpness (Zwicker acum) between
    /// the original and processed signals.
    ///
    /// Sharpness is dominated by high-frequency spectral energy, so a brightened
    /// (high-pass emphasised) copy yields a larger value while an identical copy
    /// yields ~0. Reuses [`Self::calculate_sharpness`].
    pub(crate) fn calculate_sharpness_difference(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        Ok((self.calculate_sharpness(original)? - self.calculate_sharpness(processed)?).abs())
    }

    /// Calculate sharpness (psychoacoustic measure of high-frequency content)
    pub(crate) fn calculate_sharpness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let mut sharpness = 0.0;

        // Calculate spectral centroid weighted by frequency
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            let frequency = i as f32; // Normalized frequency bin

            // Weight higher frequencies more heavily for sharpness
            let sharpness_weight = if frequency > spectrum.len() as f32 * 0.1 {
                (frequency / spectrum.len() as f32).powf(2.0) // Quadratic weighting
            } else {
                0.1 * (frequency / spectrum.len() as f32)
            };

            weighted_sum += frequency * energy * sharpness_weight;
            total_energy += energy;
        }

        if total_energy > 1e-10 {
            sharpness = weighted_sum / total_energy;
            // Normalize to approximate acum range (0-5)
            sharpness = (sharpness / spectrum.len() as f32 * 5.0).clamp(0.0, 5.0);
        }

        Ok(sharpness)
    }
}
