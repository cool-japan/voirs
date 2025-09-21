//! Harmonic processing for synthesis

use ndarray::Array1;

/// Harmonic processor
pub struct HarmonicProcessor {
    /// Harmonic amplitudes
    harmonic_amps: Vec<f32>,
    /// Harmonic phases
    harmonic_phases: Vec<f32>,
    /// Harmonic frequencies
    harmonic_freqs: Vec<f32>,
    /// Fundamental frequency
    f0: f32,
    /// Number of harmonics
    num_harmonics: usize,
    /// Sample rate
    sample_rate: f32,
    /// Phase accumulator for each harmonic
    phase_accumulators: Vec<f32>,
}

impl HarmonicProcessor {
    /// Create a new harmonic processor
    pub fn new(sample_rate: f32) -> Self {
        let num_harmonics = 32; // Default number of harmonics
        let mut harmonic_amps = vec![0.0; num_harmonics];
        let harmonic_phases = vec![0.0; num_harmonics];
        let harmonic_freqs = vec![0.0; num_harmonics];
        let phase_accumulators = vec![0.0; num_harmonics];

        // Initialize with typical harmonic amplitude decay
        for i in 0..num_harmonics {
            harmonic_amps[i] = 1.0 / (i + 1) as f32; // 1/n decay
        }

        Self {
            harmonic_amps,
            harmonic_phases,
            harmonic_freqs,
            f0: 440.0,
            num_harmonics,
            sample_rate,
            phase_accumulators,
        }
    }

    /// Process audio through harmonic synthesis
    pub fn process(&mut self, input: &Array1<f32>, sample_rate: f32) -> crate::Result<Array1<f32>> {
        self.sample_rate = sample_rate;

        // Update harmonic frequencies based on fundamental
        self.update_harmonic_frequencies();

        let mut output = input.clone();

        // Add harmonic content
        for i in 0..input.len() {
            let harmonic_sample = self.generate_harmonic_sample();
            output[i] += harmonic_sample;
        }

        Ok(output)
    }

    /// Generate a single sample of harmonic content
    fn generate_harmonic_sample(&mut self) -> f32 {
        let mut sample = 0.0;

        for h in 0..self.num_harmonics {
            if self.harmonic_freqs[h] < self.sample_rate / 2.0 {
                // Below Nyquist
                // Generate harmonic
                let harmonic_value =
                    (self.phase_accumulators[h] * 2.0 * std::f32::consts::PI).sin();
                sample += harmonic_value * self.harmonic_amps[h];

                // Update phase accumulator
                self.phase_accumulators[h] += self.harmonic_freqs[h] / self.sample_rate;
                if self.phase_accumulators[h] >= 1.0 {
                    self.phase_accumulators[h] -= 1.0;
                }
            }
        }

        sample
    }

    /// Update harmonic frequencies based on fundamental
    fn update_harmonic_frequencies(&mut self) {
        for h in 0..self.num_harmonics {
            self.harmonic_freqs[h] = self.f0 * (h + 1) as f32;
        }
    }

    /// Set fundamental frequency
    pub fn set_fundamental(&mut self, f0: f32) {
        self.f0 = f0.max(20.0).min(self.sample_rate / 2.0); // Clamp to reasonable range
        self.update_harmonic_frequencies();
    }

    /// Set harmonic amplitudes
    pub fn set_amplitudes(&mut self, amplitudes: &[f32]) {
        let count = amplitudes.len().min(self.num_harmonics);
        for i in 0..count {
            self.harmonic_amps[i] = amplitudes[i].max(0.0).min(1.0); // Clamp to 0-1
        }
    }

    /// Set harmonic phases
    pub fn set_phases(&mut self, phases: &[f32]) {
        let count = phases.len().min(self.num_harmonics);
        for i in 0..count {
            self.harmonic_phases[i] = phases[i];
            // Update phase accumulator
            self.phase_accumulators[i] = phases[i] / (2.0 * std::f32::consts::PI);
        }
    }

    /// Get fundamental frequency
    pub fn fundamental(&self) -> f32 {
        self.f0
    }

    /// Get harmonic amplitudes
    pub fn amplitudes(&self) -> &[f32] {
        &self.harmonic_amps
    }

    /// Get harmonic phases
    pub fn phases(&self) -> &[f32] {
        &self.harmonic_phases
    }

    /// Get harmonic frequencies
    pub fn frequencies(&self) -> &[f32] {
        &self.harmonic_freqs
    }

    /// Get number of harmonics
    pub fn num_harmonics(&self) -> usize {
        self.num_harmonics
    }

    /// Set number of harmonics
    pub fn set_num_harmonics(&mut self, num: usize) {
        let old_num = self.num_harmonics;
        self.num_harmonics = num.min(128); // Reasonable maximum

        // Resize vectors if needed
        if num > old_num {
            self.harmonic_amps.resize(num, 0.0);
            self.harmonic_phases.resize(num, 0.0);
            self.harmonic_freqs.resize(num, 0.0);
            self.phase_accumulators.resize(num, 0.0);

            // Initialize new harmonics with 1/n decay
            for i in old_num..num {
                self.harmonic_amps[i] = 1.0 / (i + 1) as f32;
            }
        } else if num < old_num {
            self.harmonic_amps.truncate(num);
            self.harmonic_phases.truncate(num);
            self.harmonic_freqs.truncate(num);
            self.phase_accumulators.truncate(num);
        }

        self.update_harmonic_frequencies();
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.phase_accumulators.fill(0.0);
    }

    /// Apply harmonic envelope
    pub fn set_harmonic_envelope(&mut self, envelope: &[f32]) {
        let count = envelope.len().min(self.num_harmonics);
        for i in 0..count {
            self.harmonic_amps[i] *= envelope[i].max(0.0).min(1.0);
        }
    }

    /// Calculate harmonic-to-noise ratio
    pub fn calculate_hnr(&self, audio: &[f32], window_size: usize) -> f32 {
        if audio.len() < window_size {
            return 0.0;
        }

        let mut harmonic_energy = 0.0;
        let mut total_energy = 0.0;

        // Simple HNR calculation - in practice this would be more sophisticated
        for chunk in audio.chunks(window_size) {
            let chunk_energy: f32 = chunk.iter().map(|x| x * x).sum();
            total_energy += chunk_energy;

            // Estimate harmonic content (simplified)
            let autocorr = self.calculate_autocorrelation(chunk);
            harmonic_energy += autocorr * chunk_energy;
        }

        if total_energy > 0.0 && harmonic_energy > 0.0 {
            10.0 * (harmonic_energy / (total_energy - harmonic_energy)).log10()
        } else {
            -60.0 // Very low HNR
        }
    }

    /// Calculate autocorrelation for periodicity estimation
    fn calculate_autocorrelation(&self, signal: &[f32]) -> f32 {
        let period_samples = (self.sample_rate / self.f0) as usize;
        if period_samples >= signal.len() {
            return 0.0;
        }

        let mut correlation = 0.0;
        let mut energy = 0.0;

        for i in 0..(signal.len() - period_samples) {
            correlation += signal[i] * signal[i + period_samples];
            energy += signal[i] * signal[i];
        }

        if energy > 0.0 {
            correlation / energy
        } else {
            0.0
        }
    }

    /// Apply vibrato to fundamental frequency
    pub fn apply_vibrato(&mut self, rate: f32, depth: f32, phase: f32) {
        let vibrato_value = (phase * 2.0 * std::f32::consts::PI * rate).sin();
        let modulated_f0 = self.f0 * (1.0 + depth * vibrato_value);
        self.set_fundamental(modulated_f0);
    }

    /// Generate saw wave harmonics
    pub fn set_saw_harmonics(&mut self) {
        for i in 0..self.num_harmonics {
            self.harmonic_amps[i] = 1.0 / (i + 1) as f32; // 1/n amplitude
            self.harmonic_phases[i] = 0.0;
        }
    }

    /// Generate square wave harmonics
    pub fn set_square_harmonics(&mut self) {
        for i in 0..self.num_harmonics {
            if (i + 1) % 2 == 1 {
                // Odd harmonics only
                self.harmonic_amps[i] = 1.0 / (i + 1) as f32;
            } else {
                self.harmonic_amps[i] = 0.0;
            }
            self.harmonic_phases[i] = 0.0;
        }
    }

    /// Generate triangle wave harmonics
    pub fn set_triangle_harmonics(&mut self) {
        for i in 0..self.num_harmonics {
            if (i + 1) % 2 == 1 {
                // Odd harmonics only
                let harmonic_num = i + 1;
                self.harmonic_amps[i] = 1.0 / (harmonic_num * harmonic_num) as f32;
                if (harmonic_num / 2) % 2 == 1 {
                    self.harmonic_phases[i] = std::f32::consts::PI; // Phase inversion for alternating harmonics
                } else {
                    self.harmonic_phases[i] = 0.0;
                }
            } else {
                self.harmonic_amps[i] = 0.0;
            }
        }
    }
}
