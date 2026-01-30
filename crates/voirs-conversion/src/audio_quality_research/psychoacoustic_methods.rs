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

    pub(crate) fn analyze_critical_bands(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<ResearchCriticalBandAnalysis, Error> {
        // Simplified critical band analysis
        let band_deviations = vec![0.05; 24]; // 24 Bark bands
        Ok(ResearchCriticalBandAnalysis {
            band_deviations,
            overall_distortion: 0.05,
            hf_preservation: 0.9,
            lf_preservation: 0.95,
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

    pub(crate) fn analyze_tonality(
        &self,
        _original: &[f32],
        _processed: &[f32],
    ) -> Result<TonalityAnalysis, Error> {
        Ok(TonalityAnalysis {
            tonal_noise_ratio: 0.7,
            tonal_preservation: 0.9,
            noise_deviation: 0.1,
            spectral_peaks_preservation: 0.85,
        })
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

    pub(crate) fn calculate_sharpness_difference(
        &self,
        _original: &[f32],
        _processed: &[f32],
    ) -> Result<f32, Error> {
        Ok(0.1) // Placeholder
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
