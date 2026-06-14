//! Noise gate and spectral subtraction for audio cleanup
//!
//! Implements a noise gate with configurable attack/release times and
//! spectral subtraction for advanced noise reduction.

use super::{db_to_linear, linear_to_db, NoiseGateConfig};
use crate::{AudioBuffer, Result};
use scirs2_core::Complex;

/// Noise gate with envelope following and spectral subtraction
#[derive(Debug)]
pub struct NoiseGate {
    config: NoiseGateConfig,
    sample_rate: f32,

    // Gate state
    gate_state: GateState,
    envelope: f32,
    hold_counter: u32,
    hold_samples: u32,

    // Attack/release coefficients
    attack_coeff: f32,
    release_coeff: f32,

    // Spectral subtraction
    spectral_processor: Option<SpectralSubtractor>,

    // Statistics
    current_attenuation: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GateState {
    Open,
    Closing,
    Closed,
    Opening,
    Hold,
}

impl NoiseGate {
    /// Create a new noise gate with the given configuration
    pub fn new(config: &NoiseGateConfig, sample_rate: f32) -> Result<Self> {
        let attack_coeff = (-1.0 / (config.attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (config.release_ms * 0.001 * sample_rate)).exp();
        let hold_samples = (config.hold_ms * 0.001 * sample_rate) as u32;

        let spectral_processor = if config.spectral_subtraction {
            Some(SpectralSubtractor::new(
                sample_rate,
                config.subtraction_factor,
            )?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            sample_rate,
            gate_state: GateState::Closed,
            envelope: 0.0,
            hold_counter: 0,
            hold_samples,
            attack_coeff,
            release_coeff,
            spectral_processor,
            current_attenuation: 0.0,
        })
    }

    /// Process audio buffer with noise gating
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // First apply spectral subtraction if enabled
        if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.process(audio)?;
        }

        // Then apply the noise gate
        self.apply_gate(audio)?;

        Ok(())
    }

    fn apply_gate(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();
        let threshold_linear = db_to_linear(self.config.threshold);

        for sample in samples.iter_mut() {
            // Calculate input level
            let input_level = sample.abs();

            // Update envelope
            let target_envelope = input_level;
            if target_envelope > self.envelope {
                self.envelope =
                    target_envelope + (self.envelope - target_envelope) * self.attack_coeff;
            } else {
                self.envelope =
                    target_envelope + (self.envelope - target_envelope) * self.release_coeff;
            }

            // Determine gate state based on envelope
            let should_open = self.envelope > threshold_linear;

            // State machine for gate behavior
            match self.gate_state {
                GateState::Closed => {
                    if should_open {
                        self.gate_state = GateState::Opening;
                    }
                }
                GateState::Opening => {
                    if !should_open {
                        self.gate_state = GateState::Closing;
                    } else {
                        self.gate_state = GateState::Open;
                    }
                }
                GateState::Open => {
                    if !should_open {
                        self.gate_state = GateState::Hold;
                        self.hold_counter = 0;
                    }
                }
                GateState::Hold => {
                    if should_open {
                        self.gate_state = GateState::Open;
                    } else {
                        self.hold_counter += 1;
                        if self.hold_counter >= self.hold_samples {
                            self.gate_state = GateState::Closing;
                        }
                    }
                }
                GateState::Closing => {
                    if should_open {
                        self.gate_state = GateState::Opening;
                    } else {
                        self.gate_state = GateState::Closed;
                    }
                }
            }

            // Calculate gate gain based on state
            let gate_gain = match self.gate_state {
                GateState::Open | GateState::Hold => 1.0,
                GateState::Closed => 0.0,
                GateState::Opening => {
                    // Smooth fade in
                    let fade_ratio = self.envelope / threshold_linear;
                    fade_ratio.min(1.0)
                }
                GateState::Closing => {
                    // Smooth fade out
                    let fade_ratio = self.envelope / threshold_linear;
                    fade_ratio.min(1.0)
                }
            };

            // Apply gate gain
            *sample *= gate_gain;
            self.current_attenuation = linear_to_db(gate_gain);
        }

        Ok(())
    }

    /// Get current attenuation in dB
    pub fn attenuation(&self) -> f32 {
        self.current_attenuation
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.gate_state = GateState::Closed;
        self.envelope = 0.0;
        self.hold_counter = 0;
        self.current_attenuation = 0.0;

        if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.reset();
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &NoiseGateConfig) -> Result<()> {
        let old_spectral_enabled = self.config.spectral_subtraction;
        self.config = config.clone();

        // Recalculate coefficients
        self.attack_coeff = (-1.0 / (config.attack_ms * 0.001 * self.sample_rate)).exp();
        self.release_coeff = (-1.0 / (config.release_ms * 0.001 * self.sample_rate)).exp();
        self.hold_samples = (config.hold_ms * 0.001 * self.sample_rate) as u32;

        // Update spectral processor
        if config.spectral_subtraction && !old_spectral_enabled {
            self.spectral_processor = Some(SpectralSubtractor::new(
                self.sample_rate,
                config.subtraction_factor,
            )?);
        } else if !config.spectral_subtraction {
            self.spectral_processor = None;
        } else if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.update_factor(config.subtraction_factor);
        }

        Ok(())
    }

    /// Provide a noise-only reference so spectral subtraction can build an
    /// accurate noise magnitude profile instead of estimating it from the
    /// lowest-energy frames of the processed signal.
    ///
    /// Returns `true` if a spectral-subtraction processor was present and the
    /// profile was learned, `false` otherwise.
    pub fn learn_noise_profile(&mut self, noise: &AudioBuffer) -> bool {
        if let Some(ref mut spectral_processor) = self.spectral_processor {
            spectral_processor.learn_noise_profile(noise.samples());
            true
        } else {
            false
        }
    }
}

/// Relative spectral floor coefficient (Berouti-style β floor).
///
/// Each cleaned bin is held at no less than this fraction of the **input**
/// magnitude. Flooring relative to the input (rather than the noise estimate)
/// keeps the magnitude non-negative, tames musical noise, and—crucially—keeps
/// genuine silence silent instead of injecting comfort noise.
const SPECTRAL_FLOOR_BETA: f32 = 0.02;

/// Single-channel spectral-subtraction noise reducer.
///
/// Pipeline:
/// 1. STFT analysis with a Hann window at 75% overlap (`fft_size = 1024`,
///    `hop_size = 256`) using [`scirs2_fft::rfft`].
/// 2. Per-bin magnitude / phase decomposition.
/// 3. Over-subtraction of an estimated noise magnitude spectrum with a relative
///    spectral floor.
/// 4. Magnitude recombination with the **original** phase.
/// 5. ISTFT via [`scirs2_fft::irfft`] (which carries the `1/N` factor) with
///    weighted overlap-add and window-energy normalization, preserving length.
#[derive(Debug)]
struct SpectralSubtractor {
    #[allow(dead_code)]
    sample_rate: f32,
    /// STFT frame length (FFT size).
    fft_size: usize,
    /// Analysis/synthesis hop in samples (`fft_size / 4` → 75% overlap).
    hop_size: usize,
    /// Hann window applied for both analysis and synthesis.
    window: Vec<f32>,
    /// Estimated noise magnitude spectrum (`fft_size / 2 + 1` bins).
    noise_spectrum: Vec<f32>,
    /// Base subtraction knob from configuration (drives over-subtraction).
    subtraction_factor: f32,
    /// Whether `noise_spectrum` currently holds a valid estimate.
    noise_estimated: bool,
}

impl SpectralSubtractor {
    fn new(sample_rate: f32, subtraction_factor: f32) -> Result<Self> {
        let fft_size = 1024;
        let hop_size = fft_size / 4; // 75% overlap

        // Hann window (used for both analysis and synthesis).
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        Ok(Self {
            sample_rate,
            fft_size,
            hop_size,
            window,
            noise_spectrum: vec![0.0; fft_size / 2 + 1],
            subtraction_factor,
            noise_estimated: false,
        })
    }

    /// Number of unique real-FFT bins (`fft_size / 2 + 1`).
    fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Over-subtraction factor α (≥ 1): larger values remove more noise. Mapped
    /// from the configured subtraction factor so the existing knob stays
    /// meaningful (default `0.5` → α = `2.0`).
    fn over_subtraction_factor(&self) -> f32 {
        1.0 + 2.0 * self.subtraction_factor.clamp(0.0, 2.0)
    }

    /// Average magnitude spectrum over every full Hann-windowed frame of
    /// `signal`. Shared by explicit noise-profile learning.
    fn average_magnitude_spectrum(&self, signal: &[f32]) -> Option<Vec<f32>> {
        let mut accum = vec![0.0f32; self.num_bins()];
        let mut frames = 0usize;
        let mut pos = 0;
        while pos + self.fft_size <= signal.len() {
            let mut frame = vec![0.0f32; self.fft_size];
            for (i, w) in self.window.iter().enumerate() {
                frame[i] = signal[pos + i] * w;
            }
            if let Ok(spectrum) = scirs2_fft::rfft(&frame, Some(self.fft_size)) {
                for (acc, value) in accum.iter_mut().zip(spectrum.iter()) {
                    *acc += value.norm() as f32;
                }
                frames += 1;
            }
            pos += self.hop_size;
        }
        if frames == 0 {
            return None;
        }
        for acc in &mut accum {
            *acc /= frames as f32;
        }
        Some(accum)
    }

    /// Learn the noise magnitude profile from a noise-only reference signal.
    fn learn_noise_profile(&mut self, noise: &[f32]) {
        if let Some(profile) = self.average_magnitude_spectrum(noise) {
            self.noise_spectrum = profile;
            self.noise_estimated = true;
        }
    }

    /// Estimate the noise magnitude spectrum from the lowest-energy frames of
    /// the signal itself (minimum-statistics style fallback used when no
    /// explicit noise profile has been supplied).
    fn estimate_noise_from_signal(&mut self, signal: &[f32]) {
        let bins = self.num_bins();
        let mut frame_mags: Vec<Vec<f32>> = Vec::new();
        let mut frame_energy: Vec<f32> = Vec::new();
        let mut pos = 0;
        while pos + self.fft_size <= signal.len() {
            let mut frame = vec![0.0f32; self.fft_size];
            for (i, w) in self.window.iter().enumerate() {
                frame[i] = signal[pos + i] * w;
            }
            if let Ok(spectrum) = scirs2_fft::rfft(&frame, Some(self.fft_size)) {
                let mags: Vec<f32> = spectrum.iter().map(|c| c.norm() as f32).collect();
                let energy: f32 = mags.iter().map(|m| m * m).sum();
                frame_mags.push(mags);
                frame_energy.push(energy);
            }
            pos += self.hop_size;
        }
        if frame_mags.is_empty() {
            return;
        }

        // Average the magnitude spectra of the quietest ~20% of frames.
        let mut order: Vec<usize> = (0..frame_energy.len()).collect();
        order.sort_by(|&a, &b| {
            frame_energy[a]
                .partial_cmp(&frame_energy[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let take = (order.len() / 5).max(1);
        let mut noise = vec![0.0f32; bins];
        for &fi in order.iter().take(take) {
            for (n, m) in noise.iter_mut().zip(frame_mags[fi].iter()) {
                *n += *m;
            }
        }
        for n in &mut noise {
            *n /= take as f32;
        }
        self.noise_spectrum = noise;
        self.noise_estimated = true;
    }

    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let samples = audio.samples_mut();
        let n = samples.len();

        // At least one full analysis frame is needed for STFT processing.
        if n < self.fft_size {
            return Ok(());
        }

        // Obtain a noise estimate if none was supplied explicitly.
        if !self.noise_estimated {
            let signal = samples.to_vec();
            self.estimate_noise_from_signal(&signal);
        }

        // Weighted overlap-add accumulators.
        let mut output = vec![0.0f32; n];
        let mut window_energy = vec![0.0f32; n];

        let mut pos = 0;
        while pos + self.fft_size <= n {
            // Analysis window.
            let mut frame = vec![0.0f32; self.fft_size];
            for (i, w) in self.window.iter().enumerate() {
                frame[i] = samples[pos + i] * w;
            }

            // Frequency-domain spectral subtraction.
            let processed = self.apply_spectral_subtraction(&frame);

            // Synthesis window + overlap-add, accumulating window energy for
            // exact normalization (perfect reconstruction in the no-subtraction
            // limit).
            for (i, w) in self.window.iter().enumerate() {
                output[pos + i] += processed[i] * w;
                window_energy[pos + i] += w * w;
            }
            pos += self.hop_size;
        }

        // Normalize by accumulated window energy (WOLA). Edge regions not
        // covered by any full frame keep their original samples.
        for ((sample, &out), &energy) in samples
            .iter_mut()
            .zip(output.iter())
            .zip(window_energy.iter())
        {
            if energy > 1e-6 {
                *sample = out / energy;
            }
        }

        Ok(())
    }

    /// Spectral subtraction of a single windowed analysis frame.
    ///
    /// Forward real FFT → over-subtract the estimated noise magnitude with a
    /// relative spectral floor → recombine with the **original** phase →
    /// inverse real FFT. Returns the length-`fft_size` time-domain frame. The
    /// signature is unchanged from the previous placeholder.
    fn apply_spectral_subtraction(&self, frame: &[f32]) -> Vec<f32> {
        // Forward real FFT → fft_size/2 + 1 complex bins.
        let spectrum = match scirs2_fft::rfft(frame, Some(self.fft_size)) {
            Ok(s) => s,
            Err(_) => return frame.to_vec(),
        };

        let over_subtraction = f64::from(self.over_subtraction_factor());
        let floor_beta = f64::from(SPECTRAL_FLOOR_BETA);

        let mut cleaned: Vec<Complex<f64>> = Vec::with_capacity(spectrum.len());
        for (bin, value) in spectrum.iter().enumerate() {
            let magnitude = value.norm();
            let phase = value.arg();
            let noise_mag = f64::from(*self.noise_spectrum.get(bin).unwrap_or(&0.0));

            // Over-subtract the noise magnitude, then hold at the relative
            // spectral floor so the magnitude never goes negative and silence
            // stays silent.
            let subtracted = magnitude - over_subtraction * noise_mag;
            let clean_mag = subtracted.max(floor_beta * magnitude).max(0.0);

            // Recombine the cleaned magnitude with the original phase.
            cleaned.push(Complex::new(
                clean_mag * phase.cos(),
                clean_mag * phase.sin(),
            ));
        }

        // Inverse real FFT (carries 1/N) back to the time domain.
        match scirs2_fft::irfft(&cleaned, Some(self.fft_size)) {
            Ok(time) => time.into_iter().map(|v| v as f32).collect(),
            Err(_) => frame.to_vec(),
        }
    }

    fn update_factor(&mut self, factor: f32) {
        self.subtraction_factor = factor;
    }

    fn reset(&mut self) {
        self.noise_estimated = false;
        self.noise_spectrum.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_gate_creation() {
        let config = NoiseGateConfig::default();
        let gate = NoiseGate::new(&config, 48000.0);
        assert!(gate.is_ok());
    }

    #[test]
    fn test_noise_gate_process() {
        let config = NoiseGateConfig {
            threshold: -40.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            hold_ms: 10.0,
            spectral_subtraction: false,
            subtraction_factor: 0.5,
            enabled: true,
        };

        let mut gate = NoiseGate::new(&config, 48000.0).unwrap();

        // Test with quiet signal (should be gated)
        let quiet_samples = vec![0.001; 100];
        let mut quiet_audio = AudioBuffer::from_samples(quiet_samples, 48000.0);

        let result = gate.process(&mut quiet_audio);
        assert!(result.is_ok());

        // Check that quiet signal was attenuated
        let max_amplitude = quiet_audio
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0, f32::max);
        assert!(max_amplitude < 0.001);
    }

    #[test]
    fn test_noise_gate_with_spectral_subtraction() {
        let config = NoiseGateConfig {
            spectral_subtraction: true,
            subtraction_factor: 0.3,
            ..NoiseGateConfig::default()
        };

        let gate = NoiseGate::new(&config, 48000.0);
        assert!(gate.is_ok());

        let mut gate = gate.unwrap();
        assert!(gate.spectral_processor.is_some());

        // Test processing
        let samples = vec![0.1; 2048]; // Enough samples for spectral processing
        let mut audio = AudioBuffer::from_samples(samples, 48000.0);

        let result = gate.process(&mut audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectral_subtractor() {
        let processor = SpectralSubtractor::new(48000.0, 0.5);
        assert!(processor.is_ok());

        let mut processor = processor.unwrap();

        // Test with sufficient samples
        let samples = vec![0.1; 2048];
        let mut audio = AudioBuffer::from_samples(samples, 48000.0);

        let result = processor.process(&mut audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gate_state_transitions() {
        let config = NoiseGateConfig::default();
        let mut gate = NoiseGate::new(&config, 48000.0).unwrap();

        // Initially should be closed
        assert_eq!(gate.gate_state, GateState::Closed);

        // Test with loud signal
        let loud_samples = vec![0.5; 100];
        let mut loud_audio = AudioBuffer::from_samples(loud_samples, 48000.0);

        gate.process(&mut loud_audio).unwrap();

        // Should have opened or be opening
        assert!(matches!(
            gate.gate_state,
            GateState::Open | GateState::Opening
        ));
    }

    /// Deterministic LCG producing pseudo-noise in `[-1, 1)` (no `rand`).
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use the high 24 bits for a value in [0, 1), then map to [-1, 1).
        let unit = (*state >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }

    /// Total energy of `samples` over real-FFT bins in `[lo, hi]`.
    fn band_energy(samples: &[f32], n: usize, lo: usize, hi: usize) -> f64 {
        let spectrum = scirs2_fft::rfft(samples, Some(n)).expect("rfft of band-energy probe");
        spectrum
            .iter()
            .enumerate()
            .filter(|(k, _)| *k >= lo && *k <= hi)
            .map(|(_, c)| c.norm() * c.norm())
            .sum()
    }

    #[test]
    fn test_spectral_subtraction_reduces_noise_preserves_tone() {
        let sample_rate = 16_000.0f32;
        let n = 16_384usize; // power of two so the tone lands exactly on a bin
        let tone_freq = 1_000.0f32;
        let tone_amp = 0.5f32;
        let noise_amp = 0.08f32;

        // Noise-only reference (one LCG stream) used to learn the profile.
        let mut noise_state = 0x1234_5678_9abc_def0u64;
        let noise_only: Vec<f32> = (0..n)
            .map(|_| noise_amp * lcg_next(&mut noise_state))
            .collect();

        // Tone plus an independent noise realization (different seed).
        let mut signal_state = 0x0fed_cba9_8765_4321u64;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                tone_amp * (2.0 * std::f32::consts::PI * tone_freq * t).sin()
                    + noise_amp * lcg_next(&mut signal_state)
            })
            .collect();

        let bin_of = |freq: f32| -> usize { (freq * n as f32 / sample_rate).round() as usize };
        let tone_bin = bin_of(tone_freq);
        let (noise_lo, noise_hi) = (bin_of(3_000.0), bin_of(6_000.0));

        let pre_noise = band_energy(&signal, n, noise_lo, noise_hi);
        let pre_tone = band_energy(&signal, n, tone_bin - 2, tone_bin + 2);

        let mut processor = SpectralSubtractor::new(sample_rate, 0.5).unwrap();
        processor.learn_noise_profile(&noise_only);

        let mut audio = AudioBuffer::from_samples(signal, sample_rate);
        processor.process(&mut audio).unwrap();
        let cleaned = audio.samples();

        let post_noise = band_energy(cleaned, n, noise_lo, noise_hi);
        let post_tone = band_energy(cleaned, n, tone_bin - 2, tone_bin + 2);

        // Noise-band energy must drop materially...
        assert!(
            post_noise < 0.7 * pre_noise,
            "noise band not reduced: pre={pre_noise}, post={post_noise}"
        );
        // ...while the tone bin is largely preserved.
        assert!(
            post_tone > 0.5 * pre_tone,
            "tone not preserved: pre={pre_tone}, post={post_tone}"
        );
        // Output must remain finite and length-preserving.
        assert_eq!(cleaned.len(), n);
        assert!(cleaned.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_spectral_subtraction_silence_stays_quiet() {
        let sample_rate = 16_000.0f32;
        let n = 8_192usize;
        let noise_amp = 0.1f32;

        let mut noise_state = 0xdead_beef_cafe_babeu64;
        let noise_only: Vec<f32> = (0..n)
            .map(|_| noise_amp * lcg_next(&mut noise_state))
            .collect();

        let mut processor = SpectralSubtractor::new(sample_rate, 0.5).unwrap();
        processor.learn_noise_profile(&noise_only);

        // Pure silence must stay near-zero (no comfort-noise injection).
        let mut silence = AudioBuffer::from_samples(vec![0.0f32; n], sample_rate);
        processor.process(&mut silence).unwrap();

        let max_abs = silence
            .samples()
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs.is_finite());
        assert!(max_abs < 1e-3, "silence should stay quiet, got {max_abs}");
    }

    #[test]
    fn test_noise_gate_learn_noise_profile() {
        let config = NoiseGateConfig {
            spectral_subtraction: true,
            ..NoiseGateConfig::default()
        };
        let mut gate = NoiseGate::new(&config, 16_000.0).unwrap();
        let noise = AudioBuffer::from_samples(vec![0.01f32; 4_096], 16_000.0);
        assert!(gate.learn_noise_profile(&noise));

        // With spectral subtraction disabled there is no processor to learn.
        let config_off = NoiseGateConfig {
            spectral_subtraction: false,
            ..NoiseGateConfig::default()
        };
        let mut gate_off = NoiseGate::new(&config_off, 16_000.0).unwrap();
        assert!(!gate_off.learn_noise_profile(&noise));
    }
}
