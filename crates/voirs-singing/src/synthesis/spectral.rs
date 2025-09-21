//! Spectral processing for synthesis

use ndarray::Array1;
use rustfft::{num_complex::Complex, FftPlanner};

/// Spectral processor
pub struct SpectralProcessor {
    /// FFT planner
    fft_planner: FftPlanner<f32>,
    /// Frame size
    frame_size: usize,
    /// Spectral envelope
    envelope: Array1<f32>,
    /// Formant frequencies
    formant_freqs: Vec<f32>,
    /// Formant bandwidths
    formant_bws: Vec<f32>,
    /// Formant gains
    formant_gains: Vec<f32>,
    /// FFT buffer
    fft_buffer: Vec<Complex<f32>>,
    /// IFFT buffer
    ifft_buffer: Vec<Complex<f32>>,
}

impl SpectralProcessor {
    /// Create a new spectral processor
    pub fn new(frame_size: usize) -> Self {
        let envelope = Array1::ones(frame_size / 2 + 1);
        let fft_buffer = vec![Complex::new(0.0, 0.0); frame_size];
        let ifft_buffer = vec![Complex::new(0.0, 0.0); frame_size];

        Self {
            fft_planner: FftPlanner::new(),
            frame_size,
            envelope,
            formant_freqs: vec![700.0, 1220.0, 2600.0], // Default vowel formants
            formant_bws: vec![80.0, 100.0, 150.0],
            formant_gains: vec![6.0, 4.0, 2.0],
            fft_buffer,
            ifft_buffer,
        }
    }

    /// Process audio through spectral analysis and synthesis
    pub fn process(&mut self, input: &Array1<f32>, sample_rate: f32) -> crate::Result<Array1<f32>> {
        if input.len() != self.frame_size {
            return Err(crate::Error::Processing(
                "Input frame size mismatch".to_string(),
            ));
        }

        // Copy input to FFT buffer
        for (i, &sample) in input.iter().enumerate() {
            self.fft_buffer[i] = Complex::new(sample, 0.0);
        }

        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(self.frame_size);
        fft.process(&mut self.fft_buffer);

        // Apply spectral envelope
        self.apply_spectral_envelope(sample_rate)?;

        // Apply formant filtering
        self.apply_formant_filtering(sample_rate)?;

        // Copy to IFFT buffer
        self.ifft_buffer.copy_from_slice(&self.fft_buffer);

        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(self.frame_size);
        ifft.process(&mut self.ifft_buffer);

        // Extract real part and normalize
        let mut output = Array1::zeros(self.frame_size);
        let scale = 1.0 / self.frame_size as f32;
        for (i, &complex_sample) in self.ifft_buffer.iter().enumerate() {
            output[i] = complex_sample.re * scale;
        }

        Ok(output)
    }

    /// Apply spectral envelope shaping
    fn apply_spectral_envelope(&mut self, sample_rate: f32) -> crate::Result<()> {
        let nyquist = sample_rate / 2.0;
        let bin_size = nyquist / (self.frame_size / 2) as f32;

        for i in 0..=(self.frame_size / 2) {
            let freq = i as f32 * bin_size;
            let envelope_val = self.interpolate_envelope(freq, nyquist);

            // Apply to positive frequencies
            let magnitude = self.fft_buffer[i].norm();
            let phase = self.fft_buffer[i].arg();
            self.fft_buffer[i] = Complex::from_polar(magnitude * envelope_val, phase);

            // Apply to negative frequencies (if not DC or Nyquist)
            if i > 0 && i < self.frame_size / 2 {
                let neg_i = self.frame_size - i;
                self.fft_buffer[neg_i] = Complex::from_polar(magnitude * envelope_val, -phase);
            }
        }

        Ok(())
    }

    /// Apply formant filtering
    fn apply_formant_filtering(&mut self, sample_rate: f32) -> crate::Result<()> {
        let nyquist = sample_rate / 2.0;
        let bin_size = nyquist / (self.frame_size / 2) as f32;

        for i in 0..=(self.frame_size / 2) {
            let freq = i as f32 * bin_size;
            let formant_response = self.calculate_formant_response(freq);

            // Apply formant response
            let magnitude = self.fft_buffer[i].norm();
            let phase = self.fft_buffer[i].arg();
            self.fft_buffer[i] = Complex::from_polar(magnitude * formant_response, phase);

            // Apply to negative frequencies
            if i > 0 && i < self.frame_size / 2 {
                let neg_i = self.frame_size - i;
                self.fft_buffer[neg_i] = Complex::from_polar(magnitude * formant_response, -phase);
            }
        }

        Ok(())
    }

    /// Interpolate spectral envelope at given frequency
    fn interpolate_envelope(&self, freq: f32, nyquist: f32) -> f32 {
        let normalized_freq = freq / nyquist;
        let index = normalized_freq * (self.envelope.len() - 1) as f32;
        let index_floor = index.floor() as usize;
        let index_ceil = (index_floor + 1).min(self.envelope.len() - 1);
        let frac = index - index_floor as f32;

        // Linear interpolation
        self.envelope[index_floor] * (1.0 - frac) + self.envelope[index_ceil] * frac
    }

    /// Calculate formant response at given frequency
    fn calculate_formant_response(&self, freq: f32) -> f32 {
        let mut response = 1.0;

        for i in 0..self.formant_freqs.len() {
            let formant_freq = self.formant_freqs[i];
            let bandwidth = self.formant_bws[i];
            let gain = self.formant_gains[i];

            // Resonant peak response
            let q = formant_freq / bandwidth;
            let freq_ratio = freq / formant_freq;
            let resonance =
                1.0 / ((freq_ratio - 1.0 / freq_ratio).powi(2) * q.powi(2) + 1.0).sqrt();

            response *= 1.0 + (10.0_f32.powf(gain / 20.0) - 1.0) * resonance;
        }

        response
    }

    /// Set spectral envelope
    pub fn set_envelope(&mut self, envelope: &[f32]) -> crate::Result<()> {
        if envelope.len() != self.envelope.len() {
            return Err(crate::Error::Processing(
                "Envelope size mismatch".to_string(),
            ));
        }

        for (i, &val) in envelope.iter().enumerate() {
            self.envelope[i] = val.max(0.0); // Ensure non-negative
        }

        Ok(())
    }

    /// Set formant parameters
    pub fn set_formants(&mut self, freqs: &[f32], bws: &[f32], gains: &[f32]) -> crate::Result<()> {
        if freqs.len() != bws.len() || freqs.len() != gains.len() {
            return Err(crate::Error::Processing(
                "Formant parameter array size mismatch".to_string(),
            ));
        }

        self.formant_freqs = freqs.to_vec();
        self.formant_bws = bws.to_vec();
        self.formant_gains = gains.to_vec();

        Ok(())
    }

    /// Get spectral envelope
    pub fn envelope(&self) -> &Array1<f32> {
        &self.envelope
    }

    /// Get formant frequencies
    pub fn formant_frequencies(&self) -> &[f32] {
        &self.formant_freqs
    }

    /// Get formant bandwidths
    pub fn formant_bandwidths(&self) -> &[f32] {
        &self.formant_bws
    }

    /// Get formant gains
    pub fn formant_gains(&self) -> &[f32] {
        &self.formant_gains
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.envelope.fill(1.0);
        self.fft_buffer.fill(Complex::new(0.0, 0.0));
        self.ifft_buffer.fill(Complex::new(0.0, 0.0));
    }

    /// Get frame size
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Calculate spectral centroid
    pub fn calculate_centroid(&self, spectrum: &[Complex<f32>], sample_rate: f32) -> f32 {
        let nyquist = sample_rate / 2.0;
        let bin_size = nyquist / (spectrum.len() / 2) as f32;

        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &complex_bin) in spectrum.iter().enumerate().take(spectrum.len() / 2) {
            let magnitude = complex_bin.norm();
            let freq = i as f32 * bin_size;

            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Calculate spectral rolloff
    pub fn calculate_rolloff(
        &self,
        spectrum: &[Complex<f32>],
        sample_rate: f32,
        threshold: f32,
    ) -> f32 {
        let nyquist = sample_rate / 2.0;
        let bin_size = nyquist / (spectrum.len() / 2) as f32;

        // Calculate total energy
        let total_energy: f32 = spectrum
            .iter()
            .take(spectrum.len() / 2)
            .map(|&c| c.norm_sqr())
            .sum();

        let target_energy = total_energy * threshold;
        let mut cumulative_energy = 0.0;

        for (i, &complex_bin) in spectrum.iter().enumerate().take(spectrum.len() / 2) {
            cumulative_energy += complex_bin.norm_sqr();
            if cumulative_energy >= target_energy {
                return i as f32 * bin_size;
            }
        }

        nyquist // If not found, return Nyquist frequency
    }
}
