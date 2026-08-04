//! AES (Audio Engineering Society) Standards Support
//!
//! This module implements support for various AES recommended practices
//! and standards related to audio quality evaluation and measurements.
//!
//! # Supported AES Standards
//!
//! - **AES17**: Digital audio measurement methods
//! - **AES42**: High-resolution digital audio interface
//! - **AES49**: Loudness metadata for broadcast and streaming
//! - **AES53**: Multichannel surround sound systems
//!
//! # Recommended Practices
//!
//! - Dynamic range measurement
//! - THD+N (Total Harmonic Distortion plus Noise)
//! - Frequency response
//! - Phase response
//! - Impulse response
//! - Crosstalk measurement

use super::StandardsError;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use voirs_sdk::AudioBuffer;

/// AES standards validator
pub struct AesStandards {
    /// Sample rate
    sample_rate: u32,
    /// Reference level (dB FS)
    reference_level_dbfs: f32,
}

/// AES recommended practice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AesRecommendedPractice {
    /// AES17 - Digital audio measurement methods
    Aes17,
    /// AES42 - High-resolution digital audio
    Aes42,
    /// AES49 - Loudness metadata
    Aes49,
    /// AES53 - Multichannel surround sound
    Aes53,
}

/// AES17 measurement results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aes17Measurements {
    /// Dynamic range (dB)
    pub dynamic_range_db: f32,
    /// THD+N (Total Harmonic Distortion plus Noise) (%)
    pub thd_n_percent: f32,
    /// Frequency response flatness (dB)
    pub frequency_response_flatness_db: f32,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f32,
    /// Peak level (dB FS)
    pub peak_level_dbfs: f32,
    /// RMS level (dB FS)
    pub rms_level_dbfs: f32,
    /// Crest factor (dB)
    pub crest_factor_db: f32,
    /// Compliance level
    pub compliance_level: super::ComplianceLevel,
    /// Measurement notes
    pub notes: Vec<String>,
}

/// AES49 loudness metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aes49Loudness {
    /// Integrated loudness (LUFS)
    pub integrated_loudness_lufs: f32,
    /// Loudness range (LU)
    pub loudness_range_lu: f32,
    /// Maximum true peak (dB TP)
    pub max_true_peak_dbtp: f32,
    /// Compliance with EBU R128
    pub ebu_r128_compliant: bool,
    /// Compliance with ATSC A/85
    pub atsc_a85_compliant: bool,
    /// Compliance level
    pub compliance_level: super::ComplianceLevel,
}

impl AesStandards {
    /// Create new AES standards validator
    pub fn new(sample_rate: u32) -> Result<Self, StandardsError> {
        if sample_rate < 44100 {
            return Err(StandardsError::InvalidAudioData {
                message: "Sample rate must be at least 44.1 kHz for AES standards".to_string(),
            });
        }

        Ok(Self {
            sample_rate,
            reference_level_dbfs: -20.0, // Standard reference level
        })
    }

    /// Measure AES17 compliance
    pub fn measure_aes17(&self, audio: &AudioBuffer) -> Result<Aes17Measurements, StandardsError> {
        let samples = audio.samples();
        let mut notes = Vec::new();

        // 1. Calculate dynamic range
        let dynamic_range_db = self.calculate_dynamic_range(samples)?;

        // 2. Calculate THD+N
        let thd_n_percent = self.calculate_thd_n(samples)?;

        // 3. Calculate frequency response flatness
        let frequency_response_flatness_db = self.calculate_frequency_response_flatness(samples)?;

        // 4. Calculate SNR
        let snr_db = self.calculate_snr(samples)?;

        // 5. Calculate peak and RMS levels
        let peak_level = samples
            .iter()
            .map(|&s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let peak_level_dbfs = 20.0 * peak_level.max(1e-10).log10();

        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        let rms_level_dbfs = 20.0 * rms.max(1e-10).log10();

        // 6. Calculate crest factor
        let crest_factor_db = peak_level_dbfs - rms_level_dbfs;

        // Determine compliance
        let mut compliant = true;

        if dynamic_range_db < 90.0 {
            notes.push(format!(
                "Dynamic range {} dB below recommended 90 dB",
                dynamic_range_db
            ));
            compliant = false;
        }

        if thd_n_percent > 0.01 {
            notes.push(format!(
                "THD+N {} % exceeds recommended 0.01%",
                thd_n_percent
            ));
            compliant = false;
        }

        if frequency_response_flatness_db > 0.5 {
            notes.push(format!(
                "Frequency response flatness {} dB exceeds ±0.5 dB",
                frequency_response_flatness_db
            ));
            compliant = false;
        }

        let compliance_level = if compliant {
            super::ComplianceLevel::FullyCompliant
        } else if dynamic_range_db >= 80.0 && thd_n_percent < 0.1 {
            super::ComplianceLevel::PartiallyCompliant
        } else {
            super::ComplianceLevel::NotCompliant
        };

        Ok(Aes17Measurements {
            dynamic_range_db,
            thd_n_percent,
            frequency_response_flatness_db,
            snr_db,
            peak_level_dbfs,
            rms_level_dbfs,
            crest_factor_db,
            compliance_level,
            notes,
        })
    }

    /// Measure AES49 loudness compliance
    pub fn measure_aes49_loudness(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Aes49Loudness, StandardsError> {
        let samples = audio.samples();

        // 1. Calculate integrated loudness (LUFS - ITU-R BS.1770)
        let integrated_loudness_lufs = self.calculate_integrated_loudness(samples)?;

        // 2. Calculate loudness range
        let loudness_range_lu = self.calculate_loudness_range(samples)?;

        // 3. Calculate maximum true peak
        let max_true_peak_dbtp = self.calculate_true_peak(samples)?;

        // Check EBU R128 compliance (-23 LUFS ±1 LU, -1 dB TP)
        let ebu_r128_compliant = integrated_loudness_lufs >= -24.0
            && integrated_loudness_lufs <= -22.0
            && max_true_peak_dbtp <= -1.0;

        // Check ATSC A/85 compliance (-24 LUFS ±2 LU)
        let atsc_a85_compliant =
            integrated_loudness_lufs >= -26.0 && integrated_loudness_lufs <= -22.0;

        let compliance_level = if ebu_r128_compliant && atsc_a85_compliant {
            super::ComplianceLevel::FullyCompliant
        } else if atsc_a85_compliant {
            super::ComplianceLevel::PartiallyCompliant
        } else {
            super::ComplianceLevel::NotCompliant
        };

        Ok(Aes49Loudness {
            integrated_loudness_lufs,
            loudness_range_lu,
            max_true_peak_dbtp,
            ebu_r128_compliant,
            atsc_a85_compliant,
            compliance_level,
        })
    }

    /// Calculate dynamic range
    fn calculate_dynamic_range(&self, samples: &[f32]) -> Result<f32, StandardsError> {
        // Dynamic range = difference between maximum signal and noise floor

        // Find peak signal
        let peak = samples
            .iter()
            .map(|&s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Estimate noise floor (using quiet passages)
        let mut sorted_samples: Vec<f32> = samples.iter().map(|&s| s.abs()).collect();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Noise floor is approximately the 10th percentile
        let noise_floor_idx = (sorted_samples.len() as f32 * 0.1) as usize;
        let noise_floor = sorted_samples[noise_floor_idx].max(1e-10);

        // Dynamic range in dB
        let dynamic_range = 20.0 * (peak / noise_floor).log10();

        Ok(dynamic_range.min(120.0)) // Cap at 120 dB
    }

    /// Calculate THD+N (Total Harmonic Distortion plus Noise)
    fn calculate_thd_n(&self, samples: &[f32]) -> Result<f32, StandardsError> {
        use scirs2_fft::{RealFftPlanner, RealToComplex};

        // Use FFT to analyze harmonic content
        let fft_size = 8192;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<f32> = samples.iter().take(fft_size).copied().collect();
        buffer.resize(fft_size, 0.0);

        // Apply Hann window
        for (i, sample) in buffer.iter_mut().enumerate() {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
            *sample *= window;
        }

        let mut spectrum = vec![scirs2_core::Complex::new(0.0, 0.0); fft_size / 2 + 1];
        fft.process(&mut buffer, &mut spectrum)
            .map_err(|e| StandardsError::Other(e.to_string()))?;

        // Find fundamental frequency (largest peak)
        let mut max_magnitude = 0.0f32;
        let mut fundamental_bin = 0;

        for (i, &complex) in spectrum.iter().enumerate().skip(1) {
            let magnitude = complex.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                fundamental_bin = i;
            }
        }

        // Calculate power of fundamental
        let fundamental_power = spectrum[fundamental_bin].norm_sqr();

        // Calculate power of harmonics and noise
        let mut distortion_power = 0.0f32;
        for (i, &complex) in spectrum.iter().enumerate().skip(1) {
            if i != fundamental_bin {
                distortion_power += complex.norm_sqr();
            }
        }

        // THD+N percentage
        let thd_n = if fundamental_power > 0.0 {
            ((distortion_power / fundamental_power).sqrt() * 100.0).min(100.0)
        } else {
            100.0
        };

        Ok(thd_n)
    }

    /// Calculate frequency response flatness (dB spread across the audible
    /// passband).
    ///
    /// Computes the averaged magnitude spectrum (Hann-windowed, 50%-overlapping
    /// frames, `scirs2_fft::rfft`) across `samples`, converts each passband bin
    /// (20 Hz – min(20 kHz, 0.95·Nyquist)) to dB, and reports the spread between
    /// the 95th and 5th percentile bin magnitudes. This robust-range approach
    /// (rather than raw max − min) avoids letting a single narrow spectral null or
    /// a near-silent high-frequency tail dominate the figure, standard practice
    /// for frequency-response measurements. A perfectly flat response reports
    /// `0.0` dB; real-world transducers/codecs typically show several dB of
    /// passband ripple. A true swept-sine measurement (driving the device under
    /// test directly) would be more precise than this passive spectral analysis
    /// of a single recorded/rendered signal, but this is a genuine measurement
    /// from the actual samples rather than a fixed placeholder.
    fn calculate_frequency_response_flatness(
        &self,
        samples: &[f32],
    ) -> Result<f32, StandardsError> {
        if samples.len() < 64 {
            return Err(StandardsError::InvalidAudioData {
                message: "Not enough samples to measure frequency response flatness".to_string(),
            });
        }

        let fft_size = 8192usize.min(samples.len().next_power_of_two()).max(256);
        let hop = (fft_size / 2).max(1);
        let mut summed_power = vec![0.0f64; fft_size / 2 + 1];
        let mut frame_count = 0usize;
        let mut start = 0usize;
        loop {
            let available = (samples.len() - start).min(fft_size);
            let mut buffer = vec![0.0f64; fft_size];
            for (i, slot) in buffer.iter_mut().enumerate().take(available) {
                let window = 0.5
                    - 0.5
                        * (2.0 * std::f64::consts::PI * i as f64
                            / (fft_size as f64 - 1.0).max(1.0))
                        .cos();
                *slot = f64::from(samples[start + i]) * window;
            }
            if let Ok(spectrum) = scirs2_fft::rfft(&buffer, Some(fft_size)) {
                for (k, value) in spectrum.iter().enumerate().take(summed_power.len()) {
                    summed_power[k] += value.re * value.re + value.im * value.im;
                }
                frame_count += 1;
            }
            if available < fft_size {
                break;
            }
            start += hop;
            if start >= samples.len() {
                break;
            }
        }

        if frame_count == 0 {
            return Err(StandardsError::InvalidAudioData {
                message: "Unable to compute a frequency-response spectrum".to_string(),
            });
        }

        let bin_hz = f64::from(self.sample_rate) / fft_size as f64;
        let nyquist = f64::from(self.sample_rate) / 2.0;
        let passband_high = 20_000.0f64.min(nyquist * 0.95);
        let mut magnitudes_db: Vec<f64> = summed_power
            .iter()
            .enumerate()
            .filter_map(|(k, &power)| {
                let freq = k as f64 * bin_hz;
                if !(20.0..=passband_high).contains(&freq) {
                    return None;
                }
                let mean_power = power / frame_count as f64;
                if mean_power <= 1e-20 {
                    return None;
                }
                Some(10.0 * mean_power.log10())
            })
            .collect();

        if magnitudes_db.len() < 8 {
            // Too little resolvable passband energy to characterize a response
            // (e.g. a near-silent or heavily band-limited/degenerate signal).
            return Err(StandardsError::InvalidAudioData {
                message: "Insufficient passband energy to measure frequency response flatness"
                    .to_string(),
            });
        }

        magnitudes_db.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p05_idx = (magnitudes_db.len() as f64 * 0.05) as usize;
        let p95_idx = ((magnitudes_db.len() as f64 * 0.95) as usize).min(magnitudes_db.len() - 1);
        let flatness_db = (magnitudes_db[p95_idx] - magnitudes_db[p05_idx]) as f32;

        Ok(flatness_db.max(0.0))
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, samples: &[f32]) -> Result<f32, StandardsError> {
        // Simplified SNR calculation
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Estimate noise (using variance of differences)
        let mut noise_variance = 0.0f32;
        for window in samples.windows(2) {
            let diff = window[1] - window[0];
            noise_variance += diff * diff;
        }
        noise_variance /= (samples.len() - 1) as f32;
        let noise_rms = noise_variance.sqrt();

        let snr = if noise_rms > 0.0 {
            20.0 * (rms / noise_rms).log10()
        } else {
            120.0 // Very high SNR
        };

        Ok(snr.min(120.0))
    }

    /// Calculate integrated loudness (LUFS - ITU-R BS.1770)
    fn calculate_integrated_loudness(&self, samples: &[f32]) -> Result<f32, StandardsError> {
        // Simplified LUFS calculation
        // Real implementation would include K-weighting filter

        // Calculate RMS with gating
        let block_size = (self.sample_rate as f32 * 0.4) as usize; // 400 ms blocks
        let mut block_loudnesses = Vec::new();

        for chunk in samples.chunks(block_size) {
            let rms = (chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            let loudness = -0.691 + 10.0 * rms.max(1e-10).log10(); // Simplified LUFS
            block_loudnesses.push(loudness);
        }

        // Apply absolute gating (-70 LUFS)
        let gated_loudnesses: Vec<f32> = block_loudnesses
            .into_iter()
            .filter(|&l| l > -70.0)
            .collect();

        // Calculate integrated loudness
        let integrated = if !gated_loudnesses.is_empty() {
            gated_loudnesses.iter().sum::<f32>() / gated_loudnesses.len() as f32
        } else {
            -70.0
        };

        Ok(integrated)
    }

    /// Calculate loudness range
    fn calculate_loudness_range(&self, samples: &[f32]) -> Result<f32, StandardsError> {
        // Simplified loudness range calculation
        // Real implementation would use gated loudness measurements

        let block_size = (self.sample_rate as f32 * 0.4) as usize;
        let mut block_loudnesses = Vec::new();

        for chunk in samples.chunks(block_size) {
            let rms = (chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            let loudness = -0.691 + 10.0 * rms.max(1e-10).log10();
            if loudness > -70.0 {
                block_loudnesses.push(loudness);
            }
        }

        if block_loudnesses.is_empty() {
            return Ok(0.0);
        }

        block_loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Loudness range is difference between 95th and 10th percentiles
        let p10_idx = (block_loudnesses.len() as f32 * 0.1) as usize;
        let p95_idx = (block_loudnesses.len() as f32 * 0.95) as usize;

        let loudness_range = block_loudnesses[p95_idx] - block_loudnesses[p10_idx];

        Ok(loudness_range)
    }

    /// Calculate true peak level
    fn calculate_true_peak(&self, samples: &[f32]) -> Result<f32, StandardsError> {
        // True peak requires 4x oversampling
        // For simplicity, use sample peak with small margin
        let peak = samples
            .iter()
            .map(|&s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Add 0.3 dB margin for inter-sample peaks
        let true_peak_dbfs = 20.0 * peak.max(1e-10).log10() + 0.3;

        Ok(true_peak_dbfs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes_standards_creation() {
        let aes = AesStandards::new(48000);
        assert!(aes.is_ok());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let aes = AesStandards::new(32000);
        assert!(aes.is_err());
    }

    #[test]
    fn test_aes17_measurements() {
        let aes = AesStandards::new(48000).unwrap();

        // Create test signal with varying amplitude (sine wave)
        let samples: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.3)
            .collect();
        let audio = AudioBuffer::new(samples, 48000, 1);

        let result = aes.measure_aes17(&audio);
        assert!(result.is_ok());

        let measurements = result.unwrap();
        // Allow for the fact that constant-amplitude sine may have limited dynamic range
        assert!(measurements.thd_n_percent >= 0.0);
        // SNR should be positive for a clean signal
        assert!(measurements.snr_db >= 0.0);
    }

    #[test]
    fn test_aes49_loudness() {
        let aes = AesStandards::new(48000).unwrap();
        let audio = AudioBuffer::new(vec![0.1; 48000], 48000, 1);

        let result = aes.measure_aes49_loudness(&audio);
        assert!(result.is_ok());

        let loudness = result.unwrap();
        assert!(loudness.integrated_loudness_lufs < 0.0);
        assert!(loudness.loudness_range_lu >= 0.0);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_dynamic_range_calculation() {
        let aes = AesStandards::new(48000).unwrap();

        // Test with signal that has some variation (sine wave modulated by envelope)
        let mut samples = vec![0.0; 10000];
        for (i, sample) in samples.iter_mut().enumerate() {
            let envelope = (i as f32 / 1000.0).sin().abs(); // Slow envelope
            let carrier = (i as f32 / 10.0).sin(); // Faster carrier
            *sample = carrier * envelope * 0.5;
        }

        let dr = aes.calculate_dynamic_range(&samples);
        assert!(dr.is_ok());
        // Dynamic range should be positive and finite
        let dr_val = dr.unwrap();
        assert!(dr_val > 0.0 && (0.0..=120.0).contains(&dr_val));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_thd_n_calculation() {
        let aes = AesStandards::new(48000).unwrap();

        // Test with pure sine wave (should have low THD+N)
        // Use enough samples for FFT
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();

        let thd_n = aes.calculate_thd_n(&samples);
        assert!(thd_n.is_ok());
        // THD+N should be finite and reasonable (not NaN or extremely high)
        let thd_val = thd_n.unwrap();
        assert!((0.0..=100.0).contains(&thd_val));
    }

    /// Deterministic broadband "noise" signal (no `rand` dependency) via a
    /// simple linear-congruential generator: real, non-tonal energy spread
    /// across the whole passband.
    fn lcg_noise(seed: u64, len: usize, amplitude: f32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 33) as f32 / (1u64 << 31) as f32; // in [0, 1)
                (2.0 * unit - 1.0) * amplitude
            })
            .collect()
    }

    /// A deliberately, strongly *tilted* multi-tone signal: harmonically
    /// unrelated tones spanning the passband whose amplitude halves every
    /// time the frequency doubles (a steep, unmistakable low-pass tilt), so
    /// the 5th-95th percentile magnitude spread is large by construction.
    fn tilted_multitone(sample_rate: f32, len: usize) -> Vec<f32> {
        let tones = [
            (150.0_f32, 1.0_f32),
            (300.0, 0.5),
            (600.0, 0.25),
            (1_200.0, 0.125),
            (2_400.0, 0.0625),
            (4_800.0, 0.031_25),
            (9_600.0, 0.015_625),
            (16_000.0, 0.007_812_5),
        ];
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate;
                tones
                    .iter()
                    .map(|&(freq, amp)| amp * (2.0 * std::f32::consts::PI * freq * t).sin())
                    .sum::<f32>()
                    * 0.1
            })
            .collect()
    }

    /// `calculate_frequency_response_flatness` must be a real, audio-dependent
    /// measurement, not the old hardcoded `0.3` dB placeholder: a deliberately
    /// steeply-tilted spectrum must report a materially larger dB spread than
    /// a broadband, roughly-flat-spectrum signal of the same duration.
    #[test]
    fn test_frequency_response_flatness_varies_with_spectral_shape() {
        let aes = AesStandards::new(48000).unwrap();
        let sample_rate = 48_000.0_f32;
        // Several FFT-analysis frames' worth of signal so the averaged
        // periodogram is meaningful for both cases.
        let len = 48_000 * 2;

        let flat_ish = lcg_noise(0xC0FF_EE01, len, 0.3);
        let tilted = tilted_multitone(sample_rate, len);

        let flat_db = aes
            .calculate_frequency_response_flatness(&flat_ish)
            .expect("broadband noise should yield a measurable flatness");
        let tilted_db = aes
            .calculate_frequency_response_flatness(&tilted)
            .expect("tilted multitone should yield a measurable flatness");

        assert!(
            flat_db >= 0.0,
            "flatness must be non-negative, got {flat_db}"
        );
        assert!(
            tilted_db >= 0.0,
            "flatness must be non-negative, got {tilted_db}"
        );
        assert!(
            tilted_db > flat_db,
            "a steeply-tilted spectrum ({tilted_db} dB spread) must measure a larger flatness \
             figure than a broadband, roughly-flat signal ({flat_db} dB spread) -- this must \
             depend on the actual audio, not return a fixed constant"
        );
        // The old fabricated implementation always returned exactly 0.3 dB
        // regardless of input; a genuine measurement on a signal engineered
        // for a strong tilt should clear that fixed value by a wide margin.
        assert!(
            tilted_db > 0.3,
            "a deliberately steep spectral tilt should measure well above the old fabricated \
             constant of 0.3 dB, got {tilted_db}"
        );
    }

    /// Too few samples to run even a single analysis frame must fail closed
    /// with a typed error, never silently return a placeholder number.
    #[test]
    fn test_frequency_response_flatness_too_few_samples_errors() {
        let aes = AesStandards::new(48000).unwrap();
        let tiny = vec![0.1_f32; 10];
        assert!(aes.calculate_frequency_response_flatness(&tiny).is_err());
    }

    /// The AES17 rollup's frequency-response compliance sub-check
    /// (`frequency_response_flatness_db > 0.5` dB) must be able to actually
    /// fail for real audio with a genuinely poor frequency response --
    /// disproving the old always-passes (`0.3` dB constant) behavior.
    #[test]
    fn test_aes17_flags_noncompliant_frequency_response_for_real_tilted_audio() {
        let aes = AesStandards::new(48000).unwrap();
        let sample_rate = 48_000.0_f32;
        let samples = tilted_multitone(sample_rate, 48_000 * 2);
        let audio = AudioBuffer::new(samples, 48_000, 1);

        let measurements = aes
            .measure_aes17(&audio)
            .expect("measurement should succeed for a well-formed signal");

        assert!(
            measurements.frequency_response_flatness_db > 0.5,
            "the deliberately tilted test signal should measure a flatness figure above the \
             ±0.5 dB compliance threshold, got {}",
            measurements.frequency_response_flatness_db
        );
        // Re-derives the same `> 0.5 dB` threshold `measure_aes17` applies
        // internally (see the `if frequency_response_flatness_db > 0.5`
        // compliance check above in this file) -- with the old hardcoded
        // `0.3` dB constant this branch could never be reached at all.
        assert!(
            measurements
                .notes
                .iter()
                .any(|note| note.contains("Frequency response flatness")),
            "a genuinely out-of-tolerance frequency response must produce a compliance note, \
             got notes: {:?}",
            measurements.notes
        );
        assert_ne!(
            measurements.compliance_level,
            super::super::ComplianceLevel::FullyCompliant,
            "overall AES17 compliance must not report FullyCompliant when the frequency \
             response check fails, given notes: {:?}",
            measurements.notes
        );
    }
}
