//! Spectral analysis implementation methods

use super::researcher::AudioQualityResearcher;
use super::types::HarmonicDistortionAnalysis;
use crate::Error;

impl AudioQualityResearcher {
    pub(crate) fn calculate_spectral_distortion(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_energy = original.iter().map(|&x| x * x).sum::<f32>();
        let diff_energy = original
            .iter()
            .zip(processed.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum::<f32>();

        if orig_energy > 0.0 {
            Ok((diff_energy / orig_energy).sqrt())
        } else {
            Ok(0.0)
        }
    }

    pub(crate) fn calculate_cepstral_distance(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        // Simplified cepstral distance calculation
        let orig_log_spec = self.log_magnitude_spectrum(original);
        let proc_log_spec = self.log_magnitude_spectrum(processed);

        let distance = orig_log_spec
            .iter()
            .zip(proc_log_spec.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum::<f32>()
            / orig_log_spec.len() as f32;

        Ok(distance.sqrt())
    }

    pub(crate) fn calculate_log_spectral_distance(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let mut distance = 0.0;
        for (i, (&orig, &proc)) in orig_spectrum.iter().zip(proc_spectrum.iter()).enumerate() {
            if i > 0 && orig > 1e-10 && proc > 1e-10 {
                // Skip DC and avoid log(0)
                distance += (orig.ln() - proc.ln()).powi(2);
            }
        }

        Ok((distance / orig_spectrum.len() as f32).sqrt())
    }

    pub(crate) fn calculate_itakura_saito_distortion(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        let mut distortion = 0.0;
        for (&orig, &proc) in orig_spectrum.iter().zip(proc_spectrum.iter()) {
            if orig > 1e-10 && proc > 1e-10 {
                distortion += orig / proc - (orig / proc).ln() - 1.0;
            }
        }

        Ok(distortion / orig_spectrum.len() as f32)
    }

    pub(crate) fn calculate_spectral_correlation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);
        Ok(self.calculate_correlation(&orig_spectrum, &proc_spectrum))
    }

    pub(crate) fn calculate_spectral_flatness_deviation(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        let orig_flatness = self.calculate_spectral_flatness(original);
        let proc_flatness = self.calculate_spectral_flatness(processed);
        Ok((orig_flatness - proc_flatness).abs())
    }

    pub(crate) fn analyze_harmonic_distortion(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<HarmonicDistortionAnalysis, Error> {
        let thd = self.calculate_thd(processed, sample_rate)?;
        let harmonic_ratios = self.calculate_harmonic_ratios(processed, sample_rate)?;
        let intermodulation_distortion = self.calculate_intermodulation_distortion(processed)?;
        let harmonic_to_noise_ratio =
            self.calculate_harmonic_to_noise_ratio(processed, sample_rate)?;

        Ok(HarmonicDistortionAnalysis {
            thd,
            harmonic_ratios,
            intermodulation_distortion,
            harmonic_to_noise_ratio,
        })
    }

    pub(crate) fn calculate_thd(&self, audio: &[f32], sample_rate: u32) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Calculate magnitude spectrum
        let spectrum = self.magnitude_spectrum(audio);
        let fundamental_freq = self.estimate_fundamental_frequency(audio, sample_rate);

        if fundamental_freq <= 0.0 {
            return Ok(0.0);
        }

        // Find fundamental and harmonic peaks
        let bin_size = sample_rate as f32 / audio.len() as f32;
        let fundamental_bin = (fundamental_freq / bin_size).round() as usize;

        if fundamental_bin >= spectrum.len() {
            return Ok(0.0);
        }

        let fundamental_magnitude = spectrum[fundamental_bin];
        if fundamental_magnitude <= 1e-10 {
            return Ok(0.0);
        }

        // Calculate harmonic energies (up to 10th harmonic)
        let mut harmonic_energy = 0.0;
        for harmonic in 2..=10 {
            let harmonic_bin = (harmonic as f32 * fundamental_freq / bin_size).round() as usize;
            if harmonic_bin < spectrum.len() {
                let peak_magnitude = self.find_local_peak(&spectrum, harmonic_bin, 3);
                harmonic_energy += peak_magnitude * peak_magnitude;
            }
        }

        let fundamental_energy = fundamental_magnitude * fundamental_magnitude;
        let thd = if fundamental_energy > 0.0 {
            (harmonic_energy / fundamental_energy).sqrt()
        } else {
            0.0
        };

        Ok(thd.clamp(0.0, 1.0))
    }

    pub(crate) fn calculate_harmonic_ratios(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>, Error> {
        if audio.is_empty() {
            return Ok(vec![0.0; 10]);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let fundamental_freq = self.estimate_fundamental_frequency(audio, sample_rate);

        if fundamental_freq <= 0.0 {
            return Ok(vec![0.0; 10]);
        }

        let bin_size = sample_rate as f32 / audio.len() as f32;
        let fundamental_bin = (fundamental_freq / bin_size).round() as usize;

        if fundamental_bin >= spectrum.len() {
            return Ok(vec![0.0; 10]);
        }

        let fundamental_magnitude = spectrum[fundamental_bin];
        if fundamental_magnitude <= 1e-10 {
            return Ok(vec![0.0; 10]);
        }

        let mut ratios = Vec::new();

        // Calculate ratios for harmonics 2-11 (10 ratios total)
        for harmonic in 2..=11 {
            let harmonic_bin = (harmonic as f32 * fundamental_freq / bin_size).round() as usize;
            let ratio = if harmonic_bin < spectrum.len() {
                let harmonic_magnitude = self.find_local_peak(&spectrum, harmonic_bin, 3);
                harmonic_magnitude / fundamental_magnitude
            } else {
                0.0
            };
            ratios.push(ratio.clamp(0.0, 1.0));
        }

        Ok(ratios)
    }

    pub(crate) fn calculate_intermodulation_distortion(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.len() < 1024 {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let mut total_signal_energy = 0.0;
        let mut intermod_energy = 0.0;

        // Find peaks in the spectrum
        let mut peaks = Vec::new();
        for i in 2..spectrum.len() - 2 {
            if spectrum[i] > spectrum[i - 1]
                && spectrum[i] > spectrum[i + 1]
                && spectrum[i] > spectrum[i - 2]
                && spectrum[i] > spectrum[i + 2]
                && spectrum[i] > 0.01
            {
                // Only significant peaks
                peaks.push((i, spectrum[i]));
            }
        }

        // Sort peaks by magnitude
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top peaks as signal components
        let signal_peaks = peaks.iter().take(5).collect::<Vec<_>>();

        for &(_, magnitude) in &signal_peaks {
            total_signal_energy += magnitude * magnitude;
        }

        // Look for intermodulation products (sum and difference frequencies)
        for i in 0..signal_peaks.len() {
            for j in i + 1..signal_peaks.len() {
                let f1_bin = signal_peaks[i].0;
                let f2_bin = signal_peaks[j].0;

                // Check for sum and difference frequencies
                let sum_bin = f1_bin + f2_bin;
                let diff_bin = (f1_bin as i32 - f2_bin as i32).unsigned_abs() as usize;

                if sum_bin < spectrum.len() {
                    let sum_energy = spectrum[sum_bin] * spectrum[sum_bin];
                    intermod_energy += sum_energy;
                }

                if diff_bin < spectrum.len() && diff_bin > 0 {
                    let diff_energy = spectrum[diff_bin] * spectrum[diff_bin];
                    intermod_energy += diff_energy;
                }
            }
        }

        let imd = if total_signal_energy > 0.0 {
            (intermod_energy / total_signal_energy).sqrt()
        } else {
            0.0
        };

        Ok(imd.clamp(0.0, 1.0))
    }

    pub(crate) fn calculate_harmonic_to_noise_ratio(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);
        let fundamental_freq = self.estimate_fundamental_frequency(audio, sample_rate);

        if fundamental_freq <= 0.0 {
            return Ok(0.0);
        }

        let bin_size = sample_rate as f32 / audio.len() as f32;
        let mut harmonic_energy = 0.0;
        let mut total_energy = 0.0;

        // Calculate total spectrum energy
        for &magnitude in &spectrum {
            total_energy += magnitude * magnitude;
        }

        // Calculate harmonic energy (fundamental + harmonics)
        for harmonic in 1..=10 {
            let harmonic_bin = (harmonic as f32 * fundamental_freq / bin_size).round() as usize;
            if harmonic_bin < spectrum.len() {
                let peak_magnitude = self.find_local_peak(&spectrum, harmonic_bin, 2);
                harmonic_energy += peak_magnitude * peak_magnitude;
            }
        }

        let noise_energy = total_energy - harmonic_energy;

        let hnr_linear = if noise_energy > 1e-10 {
            harmonic_energy / noise_energy
        } else {
            1000.0 // Very high ratio
        };

        // Convert to dB
        let hnr_db = if hnr_linear > 0.0 {
            10.0 * hnr_linear.log10()
        } else {
            -60.0
        };

        Ok(hnr_db.clamp(-60.0, 60.0))
    }
}
