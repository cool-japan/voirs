//! Common helper methods for audio quality analysis

use super::researcher::AudioQualityResearcher;

impl AudioQualityResearcher {
    // Spectral analysis helpers
    pub(crate) fn magnitude_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified magnitude spectrum calculation
        let n = audio.len();
        let mut spectrum = vec![0.0; n / 2 + 1];

        for (k, spectrum_value) in spectrum.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in audio.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *spectrum_value = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    pub(crate) fn log_magnitude_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        self.magnitude_spectrum(audio)
            .iter()
            .map(|&x| if x > 1e-10 { x.ln() } else { -23.0 }) // Avoid log(0)
            .collect()
    }

    pub(crate) fn calculate_correlation(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mean_a = a.iter().sum::<f32>() / a.len() as f32;
        let mean_b = b.iter().sum::<f32>() / b.len() as f32;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for i in 0..a.len() {
            let dev_a = a[i] - mean_a;
            let dev_b = b[i] - mean_b;
            numerator += dev_a * dev_b;
            sum_sq_a += dev_a * dev_a;
            sum_sq_b += dev_b * dev_b;
        }

        if sum_sq_a == 0.0 || sum_sq_b == 0.0 {
            return 1.0; // Perfect correlation for constant signals
        }

        (numerator / (sum_sq_a * sum_sq_b).sqrt()).clamp(-1.0, 1.0)
    }

    pub(crate) fn calculate_spectral_flatness(&self, audio: &[f32]) -> f32 {
        let spectrum = self.magnitude_spectrum(audio);
        let geometric_mean = spectrum
            .iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| x.ln())
            .sum::<f32>()
            / spectrum.len() as f32;
        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 1e-10 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    pub(crate) fn calculate_envelope(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 256;
        let mut envelope = Vec::new();

        for chunk in audio.chunks(window_size) {
            let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            envelope.push(rms);
        }

        envelope
    }

    pub(crate) fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (audio.len() - 1) as f32
    }

    pub(crate) fn calculate_instantaneous_phase(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified instantaneous phase calculation
        let mut phases = Vec::new();
        for i in 1..audio.len() {
            let phase = if audio[i - 1] != 0.0 {
                (audio[i] / audio[i - 1]).atan()
            } else {
                0.0
            };
            phases.push(phase);
        }
        phases
    }

    pub(crate) fn calculate_energy_envelope(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 512;
        let mut envelope = Vec::new();

        for chunk in audio.chunks(window_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>();
            envelope.push(energy);
        }

        envelope
    }

    /// Helper method to estimate fundamental frequency using autocorrelation
    pub(crate) fn estimate_fundamental_frequency(&self, audio: &[f32], sample_rate: u32) -> f32 {
        if audio.len() < 128 {
            return 0.0;
        }

        let max_lag = (sample_rate as usize / 50).min(audio.len() / 2); // Min 50 Hz
        let min_lag = (sample_rate as usize / 800).max(8); // Max 800 Hz

        let mut max_correlation = 0.0;
        let mut best_lag = 0;

        // Autocorrelation-based pitch detection
        for lag in min_lag..max_lag {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            let samples_to_check = (audio.len() - lag).min(1024);

            for j in 0..samples_to_check {
                correlation += audio[j] * audio[j + lag];
                norm1 += audio[j] * audio[j];
                norm2 += audio[j + lag] * audio[j + lag];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation /= (norm1 * norm2).sqrt();
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        if max_correlation > 0.3 && best_lag > 0 {
            sample_rate as f32 / best_lag as f32
        } else {
            0.0
        }
    }

    /// Helper method to find local peak around a given bin
    pub(crate) fn find_local_peak(
        &self,
        spectrum: &[f32],
        center_bin: usize,
        radius: usize,
    ) -> f32 {
        let start = center_bin.saturating_sub(radius);
        let end = (center_bin + radius + 1).min(spectrum.len());

        spectrum[start..end]
            .iter()
            .fold(0.0, |max_val, &val| max_val.max(val))
    }

    /// Helper methods for multidimensional quality calculations
    pub(crate) fn detect_spectral_artifacts(&self, spectrum: &[f32]) -> f32 {
        let mut artifact_score = 0.0;

        // Look for sudden spectral peaks (potential artifacts)
        for i in 2..spectrum.len() - 2 {
            let current = spectrum[i];
            let neighbors =
                (spectrum[i - 2] + spectrum[i - 1] + spectrum[i + 1] + spectrum[i + 2]) / 4.0;

            if current > neighbors * 3.0 && current > 0.01 {
                artifact_score += current - neighbors;
            }
        }

        (artifact_score * 10.0).clamp(0.0, 1.0)
    }

    pub(crate) fn calculate_dynamic_range(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let envelope = self.calculate_envelope(audio);
        if envelope.is_empty() {
            return 0.0;
        }

        let max_val = envelope.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = envelope.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        if max_val > min_val && min_val > 1e-10 {
            ((max_val / min_val).log10() / 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub(crate) fn calculate_spectral_smoothness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.len() < 3 {
            return 1.0;
        }

        let mut smoothness_sum = 0.0;
        for i in 1..spectrum.len() - 1 {
            let variation = (spectrum[i] - (spectrum[i - 1] + spectrum[i + 1]) / 2.0).abs();
            smoothness_sum += variation;
        }

        let avg_variation = smoothness_sum / (spectrum.len() - 2) as f32;
        let avg_magnitude = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if avg_magnitude > 1e-10 {
            (1.0 - (avg_variation / avg_magnitude).min(1.0)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    pub(crate) fn analyze_decay_characteristics(&self, envelope: &[f32]) -> f32 {
        if envelope.len() < 10 {
            return 0.0;
        }

        // Look for exponential decay characteristics
        let peak_idx = envelope
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if peak_idx >= envelope.len() - 5 {
            return 0.0;
        }

        let decay_portion = &envelope[peak_idx..];
        if decay_portion.len() < 5 {
            return 0.0;
        }

        // Calculate decay rate
        let mut decay_score = 0.0;
        let peak_val = decay_portion[0];

        if peak_val > 1e-10 {
            for (i, &val) in decay_portion.iter().enumerate().skip(1) {
                let expected_decay = peak_val * (-0.1 * i as f32).exp();
                let decay_match = 1.0 - ((val - expected_decay) / peak_val).abs();
                decay_score += decay_match.max(0.0);
            }
            decay_score /= (decay_portion.len() - 1) as f32;
        }

        decay_score.clamp(0.0, 1.0)
    }

    pub(crate) fn calculate_spectral_spread(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            weighted_sum += i as f32 * energy;
            total_energy += energy;
        }

        let centroid = if total_energy > 1e-10 {
            weighted_sum / total_energy
        } else {
            return 0.0;
        };

        let mut spread_sum = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            let energy = magnitude * magnitude;
            let deviation = (i as f32 - centroid).powi(2);
            spread_sum += deviation * energy;
        }

        let spread = if total_energy > 1e-10 {
            (spread_sum / total_energy).sqrt()
        } else {
            0.0
        };

        (spread / spectrum.len() as f32).clamp(0.0, 1.0)
    }

    pub(crate) fn calculate_harmonic_warmth(&self, spectrum: &[f32]) -> f32 {
        let mut warmth_score = 0.0;
        let mut harmonic_count = 0;

        // Look for even harmonics in low-mid frequency range
        let warmth_range = spectrum.len() / 3;

        for i in (2..warmth_range).step_by(2) {
            // Even harmonics
            if i < spectrum.len() {
                warmth_score += spectrum[i] * spectrum[i];
                harmonic_count += 1;
            }
        }

        if harmonic_count > 0 {
            (warmth_score / harmonic_count as f32 * 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub(crate) fn calculate_brightness_harmonics(&self, spectrum: &[f32]) -> f32 {
        let mut brightness_score = 0.0;
        let brightness_start = spectrum.len() / 3;
        let brightness_end = spectrum.len() * 2 / 3;

        // Look for harmonic content in brightness range
        for i in brightness_start..brightness_end {
            if i > 0 && i < spectrum.len() - 1 {
                // Look for peaks (harmonics)
                if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
                    brightness_score += spectrum[i] * spectrum[i];
                }
            }
        }

        let range_size = brightness_end - brightness_start;
        if range_size > 0 {
            (brightness_score / range_size as f32 * 5.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub(crate) fn calculate_attack_definition(&self, audio: &[f32]) -> f32 {
        if audio.len() < 100 {
            return 0.0;
        }

        let envelope = self.calculate_envelope(audio);
        if envelope.len() < 10 {
            return 0.0;
        }

        // Look for fast attack characteristics
        let mut max_attack_rate = 0.0;

        for i in 1..envelope.len().min(20) {
            // Check first 20 frames for attack
            let attack_rate = envelope[i] - envelope[i - 1];
            if attack_rate > max_attack_rate {
                max_attack_rate = attack_rate;
            }
        }

        // Normalize attack rate
        let max_envelope = envelope.iter().fold(0.0f32, |a, &b| a.max(b));

        if max_envelope > 1e-10 {
            (max_attack_rate / max_envelope * 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
