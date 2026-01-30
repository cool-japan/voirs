//! Multidimensional quality analysis implementation methods

use super::researcher::AudioQualityResearcher;
use crate::Error;

impl AudioQualityResearcher {
    pub(crate) fn calculate_naturalness(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<f32, Error> {
        if original.len() != processed.len() || original.is_empty() {
            return Ok(0.0);
        }

        // Naturalness based on spectral similarity and harmonic structure preservation
        let orig_spectrum = self.magnitude_spectrum(original);
        let proc_spectrum = self.magnitude_spectrum(processed);

        // Calculate spectral correlation
        let spectral_correlation = self.calculate_correlation(&orig_spectrum, &proc_spectrum);

        // Check harmonic structure preservation
        let orig_fundamental = self.estimate_fundamental_frequency(original, sample_rate);
        let proc_fundamental = self.estimate_fundamental_frequency(processed, sample_rate);

        let fundamental_preservation = if orig_fundamental > 0.0 && proc_fundamental > 0.0 {
            let freq_ratio =
                (proc_fundamental / orig_fundamental).max(orig_fundamental / proc_fundamental);
            (2.0 - freq_ratio).clamp(0.0, 1.0)
        } else {
            0.5 // Neutral score for non-tonal signals
        };

        // Check for artifacts using spectral irregularities
        let artifact_penalty = self.detect_spectral_artifacts(&proc_spectrum);

        // Combine factors for naturalness score
        let naturalness = (spectral_correlation * 0.4
            + fundamental_preservation * 0.3
            + (1.0 - artifact_penalty) * 0.3)
            .clamp(0.0, 1.0);

        Ok(naturalness)
    }

    pub(crate) fn calculate_clarity(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Clarity based on spectral definition and high-frequency content
        let mut hf_energy = 0.0;
        let mut total_energy = 0.0;
        let hf_start = spectrum.len() * 3 / 8; // Above ~3kHz equivalent

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;

            if i >= hf_start {
                hf_energy += energy;
            }
        }

        // High-frequency to total energy ratio (clarity indicator)
        let hf_ratio = if total_energy > 1e-10 {
            hf_energy / total_energy
        } else {
            0.0
        };

        // Spectral flatness (measure of noise vs tonal content)
        let spectral_flatness = self.calculate_spectral_flatness(audio);

        // Dynamic range (clarity through contrast)
        let dynamic_range = self.calculate_dynamic_range(audio);

        // Combine factors for clarity score
        let clarity = (hf_ratio * 0.4 + (1.0 - spectral_flatness) * 0.3 + dynamic_range * 0.3)
            .clamp(0.0, 1.0);

        Ok(clarity)
    }

    pub(crate) fn calculate_pleasantness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Pleasantness based on harmonic content, smoothness, and absence of harsh artifacts
        let spectrum = self.magnitude_spectrum(audio);

        // Calculate spectral smoothness (less variation = more pleasant)
        let spectral_smoothness = self.calculate_spectral_smoothness(&spectrum);

        // Calculate roughness (lower roughness = more pleasant)
        let roughness = self.calculate_roughness(audio)?;

        // Check for harsh high-frequency content
        let hf_start = spectrum.len() * 5 / 8; // Above ~5kHz equivalent
        let mut hf_peaks = 0;

        for i in hf_start..spectrum.len() {
            if i > 0
                && i < spectrum.len() - 1
                && spectrum[i] > spectrum[i - 1] * 2.0
                && spectrum[i] > spectrum[i + 1] * 2.0
            {
                hf_peaks += 1;
            }
        }

        let harsh_hf_penalty = (hf_peaks as f32 / 10.0).min(0.3);

        // Dynamic range balance (moderate dynamics are more pleasant)
        let dynamic_range = self.calculate_dynamic_range(audio);
        let dynamic_pleasantness = if dynamic_range > 0.7 {
            1.0 - (dynamic_range - 0.7) // Penalize excessive dynamics
        } else {
            dynamic_range / 0.7 // Reward up to moderate dynamics
        };

        // Combine factors for pleasantness score
        let pleasantness = (spectral_smoothness * 0.3
            + (1.0 - roughness) * 0.3
            + dynamic_pleasantness * 0.2
            + (1.0 - harsh_hf_penalty) * 0.2)
            .clamp(0.0, 1.0);

        Ok(pleasantness)
    }

    pub(crate) fn calculate_intelligibility(
        &self,
        original: &[f32],
        processed: &[f32],
    ) -> Result<f32, Error> {
        // Use correlation as a simple intelligibility measure
        Ok(self.calculate_correlation(original, processed))
    }

    pub(crate) fn calculate_spaciousness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Spaciousness based on reverb characteristics and stereo width
        let envelope = self.calculate_envelope(audio);

        // Calculate decay characteristics (indicative of space)
        let decay_score = self.analyze_decay_characteristics(&envelope);

        // Calculate spectral diffusion (wider spectrum = more spacious)
        let spectrum = self.magnitude_spectrum(audio);
        let spectral_spread = self.calculate_spectral_spread(&spectrum);

        // Late reflection simulation (longer envelope tails = more spacious)
        let late_reflection_score = if envelope.len() > 10 {
            let tail_start = envelope.len() * 3 / 4;
            let tail_energy: f32 = envelope[tail_start..].iter().sum();
            let total_energy: f32 = envelope.iter().sum();

            if total_energy > 1e-10 {
                (tail_energy / total_energy * 4.0).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Combine factors for spaciousness score
        let spaciousness =
            (decay_score * 0.4 + spectral_spread * 0.3 + late_reflection_score * 0.3)
                .clamp(0.0, 1.0);

        Ok(spaciousness)
    }

    pub(crate) fn calculate_warmth(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Warmth based on low-frequency content and harmonic richness
        let mut lf_energy = 0.0;
        let mut mf_energy = 0.0;
        let mut total_energy = 0.0;

        let lf_cutoff = spectrum.len() / 8; // Low frequencies
        let mf_cutoff = spectrum.len() / 3; // Mid frequencies

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;

            if i < lf_cutoff {
                lf_energy += energy;
            } else if i < mf_cutoff {
                mf_energy += energy;
            }
        }

        // Low to mid frequency ratio (warmth indicator)
        let lf_mf_ratio = if mf_energy > 1e-10 {
            (lf_energy / mf_energy).min(2.0) / 2.0
        } else {
            0.0
        };

        // Overall low frequency content
        let lf_content = if total_energy > 1e-10 {
            lf_energy / total_energy
        } else {
            0.0
        };

        // Harmonic warmth (even harmonics contribute to warmth)
        let harmonic_warmth = self.calculate_harmonic_warmth(&spectrum);

        // Combine factors for warmth score
        let warmth = (lf_mf_ratio * 0.4 + lf_content * 0.3 + harmonic_warmth * 0.3).clamp(0.0, 1.0);

        Ok(warmth)
    }

    pub(crate) fn calculate_brightness(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Brightness based on high-frequency content and spectral centroid
        let mut hf_energy = 0.0;
        let mut total_energy = 0.0;
        let mut weighted_freq_sum = 0.0;

        let hf_start = spectrum.len() / 3; // Above ~2.7kHz equivalent

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;
            weighted_freq_sum += i as f32 * energy;

            if i >= hf_start {
                hf_energy += energy;
            }
        }

        // High-frequency content
        let hf_ratio = if total_energy > 1e-10 {
            hf_energy / total_energy
        } else {
            0.0
        };

        // Spectral centroid (brightness indicator)
        let spectral_centroid = if total_energy > 1e-10 {
            weighted_freq_sum / total_energy / spectrum.len() as f32
        } else {
            0.0
        };

        // Presence of harmonics in the brightness range
        let brightness_harmonics = self.calculate_brightness_harmonics(&spectrum);

        // Combine factors for brightness score
        let brightness =
            (hf_ratio * 0.4 + spectral_centroid * 0.3 + brightness_harmonics * 0.3).clamp(0.0, 1.0);

        Ok(brightness)
    }

    pub(crate) fn calculate_presence(&self, audio: &[f32]) -> Result<f32, Error> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let spectrum = self.magnitude_spectrum(audio);

        // Presence based on upper-midrange content and clarity
        let mut presence_energy = 0.0;
        let mut total_energy = 0.0;

        // Presence frequency range (roughly 2-8 kHz equivalent)
        let presence_start = spectrum.len() / 4;
        let presence_end = spectrum.len() * 2 / 3;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            total_energy += energy;

            if i >= presence_start && i < presence_end {
                presence_energy += energy;
            }
        }

        // Presence frequency content
        let presence_ratio = if total_energy > 1e-10 {
            presence_energy / total_energy
        } else {
            0.0
        };

        // Dynamic range in presence region (clarity of presence)
        let presence_spectrum = &spectrum[presence_start..presence_end];
        let presence_dynamic_range = if !presence_spectrum.is_empty() {
            let max_val = presence_spectrum.iter().fold(0.0f32, |a, &b| a.max(b));
            let min_val = presence_spectrum
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            if max_val > min_val && min_val > 1e-10 {
                (max_val / min_val).log10() / 3.0 // Normalize to ~0-1
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Attack characteristics (presence through transient definition)
        let attack_definition = self.calculate_attack_definition(audio);

        // Combine factors for presence score
        let presence = (presence_ratio * 0.4
            + presence_dynamic_range.clamp(0.0, 1.0) * 0.3
            + attack_definition * 0.3)
            .clamp(0.0, 1.0);

        Ok(presence)
    }
}
