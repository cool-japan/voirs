//! Emotion feature extraction
//!
//! Extracts acoustic features relevant for emotion and sentiment recognition
//! including prosodic, spectral, and voice quality features.

use crate::RecognitionError;
use scirs2_core::numeric::Complex64;
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Emotion feature extractor
pub struct EmotionFeatureExtractor {
    /// Sample rate for feature extraction
    sample_rate: u32,
    /// Frame size for analysis
    frame_size: usize,
    /// Hop size for overlapping frames
    hop_size: usize,
}

impl EmotionFeatureExtractor {
    /// Create new feature extractor
    pub fn new() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 1024,
            hop_size: 512,
        }
    }

    /// Extract comprehensive emotion features from audio
    pub async fn extract_emotion_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let samples = audio.samples();
        let mut features = HashMap::new();

        // Prosodic features
        let prosodic_features = self.extract_prosodic_features(samples)?;
        features.extend(prosodic_features);

        // Spectral features
        let spectral_features = self.extract_spectral_features(samples)?;
        features.extend(spectral_features);

        // Voice quality features
        let voice_quality_features = self.extract_voice_quality_features(samples)?;
        features.extend(voice_quality_features);

        // Temporal features
        let temporal_features = self.extract_temporal_features(samples)?;
        features.extend(temporal_features);

        Ok(features)
    }

    /// Extract prosodic features (pitch, intensity, rhythm)
    fn extract_prosodic_features(
        &self,
        samples: &[f32],
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Fundamental frequency (F0) features
        let f0_contour = self.extract_f0_contour(samples)?;
        features.insert("f0_mean".to_string(), self.mean(&f0_contour));
        features.insert("f0_std".to_string(), self.std_dev(&f0_contour));
        features.insert("f0_min".to_string(), self.min(&f0_contour));
        features.insert("f0_max".to_string(), self.max(&f0_contour));
        features.insert(
            "f0_range".to_string(),
            self.max(&f0_contour) - self.min(&f0_contour),
        );

        // Pitch variance and trends
        features.insert("pitch_variance".to_string(), self.variance(&f0_contour));
        features.insert("pitch_slope".to_string(), self.linear_trend(&f0_contour));

        // Energy/intensity features
        let energy_contour = self.extract_energy_contour(samples)?;
        features.insert("energy_mean".to_string(), self.mean(&energy_contour));
        features.insert("energy_std".to_string(), self.std_dev(&energy_contour));
        features.insert(
            "energy_variance".to_string(),
            self.variance(&energy_contour),
        );
        features.insert(
            "energy_range".to_string(),
            self.max(&energy_contour) - self.min(&energy_contour),
        );

        // Speaking rate estimation
        features.insert(
            "speaking_rate".to_string(),
            self.estimate_speaking_rate(samples)?,
        );

        // Rhythm and timing features
        let pause_info = self.analyze_pauses(samples)?;
        features.insert("pause_frequency".to_string(), pause_info.0);
        features.insert("pause_duration_mean".to_string(), pause_info.1);

        Ok(features)
    }

    /// Extract spectral features
    fn extract_spectral_features(
        &self,
        samples: &[f32],
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Spectral centroid
        let spectral_centroid = self.compute_spectral_centroid(samples)?;
        features.insert("spectral_centroid".to_string(), spectral_centroid);

        // Spectral rolloff
        let spectral_rolloff = self.compute_spectral_rolloff(samples)?;
        features.insert("spectral_rolloff".to_string(), spectral_rolloff);

        // Spectral bandwidth
        let spectral_bandwidth = self.compute_spectral_bandwidth(samples)?;
        features.insert("spectral_bandwidth".to_string(), spectral_bandwidth);

        // Zero crossing rate
        let zcr = self.compute_zero_crossing_rate(samples);
        features.insert("zero_crossing_rate".to_string(), zcr);

        // MFCC features (first 13 coefficients)
        let mfcc = self.compute_mfcc(samples)?;
        for (i, &coeff) in mfcc.iter().take(13).enumerate() {
            features.insert(format!("mfcc_{}", i), coeff);
        }

        // Formant frequencies
        let formants = self.extract_formants(samples)?;
        for (i, &formant) in formants.iter().take(3).enumerate() {
            features.insert(format!("formant_{}", i + 1), formant);
        }

        Ok(features)
    }

    /// Extract voice quality features
    fn extract_voice_quality_features(
        &self,
        samples: &[f32],
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Jitter (pitch period irregularity)
        let jitter = self.compute_jitter(samples)?;
        features.insert("jitter".to_string(), jitter);

        // Shimmer (amplitude irregularity)
        let shimmer = self.compute_shimmer(samples)?;
        features.insert("shimmer".to_string(), shimmer);

        // Harmonic-to-noise ratio
        let hnr = self.compute_hnr(samples)?;
        features.insert("hnr".to_string(), hnr);

        // Breathiness measure
        let breathiness = self.compute_breathiness(samples)?;
        features.insert("breathiness".to_string(), breathiness);

        // Vocal fry detection
        let vocal_fry = self.detect_vocal_fry(samples)?;
        features.insert("vocal_fry".to_string(), vocal_fry);

        // Creakiness measure
        let creakiness = self.compute_creakiness(samples)?;
        features.insert("creakiness".to_string(), creakiness);

        Ok(features)
    }

    /// Extract temporal features
    fn extract_temporal_features(
        &self,
        samples: &[f32],
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Total duration
        let duration = samples.len() as f32 / self.sample_rate as f32;
        features.insert("duration".to_string(), duration);

        // Speech/silence ratio
        let speech_ratio = self.compute_speech_ratio(samples)?;
        features.insert("speech_ratio".to_string(), speech_ratio);

        // Articulation rate
        let articulation_rate = self.compute_articulation_rate(samples)?;
        features.insert("articulation".to_string(), articulation_rate);

        // Tempo variation
        let tempo_variation = self.compute_tempo_variation(samples)?;
        features.insert("tempo_variation".to_string(), tempo_variation);

        Ok(features)
    }

    /// Extract F0 contour (simplified implementation)
    fn extract_f0_contour(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let mut f0_values = Vec::new();
        let frame_count = (samples.len() - self.frame_size) / self.hop_size + 1;

        for i in 0..frame_count {
            let start = i * self.hop_size;
            let end = (start + self.frame_size).min(samples.len());
            let frame = &samples[start..end];

            // Simplified autocorrelation-based F0 estimation
            let f0 = self.estimate_f0_autocorr(frame);
            f0_values.push(f0);
        }

        Ok(f0_values)
    }

    /// Estimate F0 using autocorrelation (simplified)
    fn estimate_f0_autocorr(&self, frame: &[f32]) -> f32 {
        if frame.len() < 100 {
            return 0.0;
        }

        let mut max_corr = 0.0;
        let mut best_lag = 0;

        // Search for fundamental period
        for lag in 50..300 {
            // Typical F0 range: 50-400 Hz at 16kHz
            if lag >= frame.len() {
                break;
            }

            let mut corr = 0.0;
            for i in 0..(frame.len() - lag) {
                corr += frame[i] * frame[i + lag];
            }

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        if best_lag > 0 {
            self.sample_rate as f32 / best_lag as f32
        } else {
            0.0
        }
    }

    /// Extract energy contour
    fn extract_energy_contour(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let mut energy_values = Vec::new();
        let frame_count = (samples.len() - self.frame_size) / self.hop_size + 1;

        for i in 0..frame_count {
            let start = i * self.hop_size;
            let end = (start + self.frame_size).min(samples.len());
            let frame = &samples[start..end];

            let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            energy_values.push(energy.sqrt());
        }

        Ok(energy_values)
    }

    /// Estimate speaking rate (simplified)
    fn estimate_speaking_rate(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Count energy peaks as syllable approximation
        let energy_contour = self.extract_energy_contour(samples)?;
        let mean_energy = self.mean(&energy_contour);
        let threshold = mean_energy * 0.7;

        let mut peak_count = 0;
        let mut in_peak = false;

        for &energy in &energy_contour {
            if energy > threshold && !in_peak {
                peak_count += 1;
                in_peak = true;
            } else if energy <= threshold {
                in_peak = false;
            }
        }

        let duration = samples.len() as f32 / self.sample_rate as f32;
        Ok(peak_count as f32 / duration) // Syllables per second
    }

    /// Analyze pauses in speech
    fn analyze_pauses(&self, samples: &[f32]) -> Result<(f32, f32), RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        let mean_energy = self.mean(&energy_contour);
        let silence_threshold = mean_energy * 0.1;

        let mut pause_durations = Vec::new();
        let mut current_pause_length = 0;
        let frame_duration = self.hop_size as f32 / self.sample_rate as f32;

        for &energy in &energy_contour {
            if energy < silence_threshold {
                current_pause_length += 1;
            } else if current_pause_length > 0 {
                let pause_duration = current_pause_length as f32 * frame_duration;
                if pause_duration > 0.1 {
                    // Minimum pause duration
                    pause_durations.push(pause_duration);
                }
                current_pause_length = 0;
            }
        }

        let duration = samples.len() as f32 / self.sample_rate as f32;
        let pause_frequency = pause_durations.len() as f32 / duration;
        let mean_pause_duration = if pause_durations.is_empty() {
            0.0
        } else {
            self.mean(&pause_durations)
        };

        Ok((pause_frequency, mean_pause_duration))
    }

    // -------------------------------------------------------------------------
    // Shared FFT helper
    // -------------------------------------------------------------------------

    /// Compute Hann-windowed magnitude spectrum using rfft.
    ///
    /// Returns `(magnitudes, n_fft)` where `n_fft` is the FFT size used and
    /// `magnitudes` has length `n_fft / 2 + 1`.
    fn compute_windowed_spectrum(
        &self,
        samples: &[f32],
    ) -> Result<(Vec<f64>, usize), RecognitionError> {
        // Choose n as next power of two of the signal length, clamped to [128, 4096]
        let n = samples.len().next_power_of_two().min(4096).max(128);

        let mut buf = vec![0.0f64; n];
        for (i, &s) in samples.iter().take(n).enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
            buf[i] = s as f64 * w;
        }

        let spectrum: Vec<Complex64> =
            scirs2_fft::rfft(&buf, Some(n)).map_err(|e| RecognitionError::AudioAnalysisError {
                message: format!("FFT failed in feature extraction: {:?}", e),
                source: None,
            })?;

        let mags: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        Ok((mags, n))
    }

    // -------------------------------------------------------------------------
    // Spectral centroid – real FFT implementation
    // -------------------------------------------------------------------------

    /// Compute spectral centroid in Hz.
    ///
    /// Uses a Hann-windowed rfft: centroid = Σ(k * freq_res * |X[k]|) / Σ|X[k]|
    fn compute_spectral_centroid(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let (mags, n) = self.compute_windowed_spectrum(samples)?;
        let freq_resolution = self.sample_rate as f64 / n as f64;

        let mut weighted_sum = 0.0f64;
        let mut magnitude_sum = 0.0f64;

        for (k, &mag) in mags.iter().enumerate() {
            let freq_hz = k as f64 * freq_resolution;
            weighted_sum += freq_hz * mag;
            magnitude_sum += mag;
        }

        if magnitude_sum > 1e-12 {
            Ok((weighted_sum / magnitude_sum) as f32)
        } else {
            Ok(0.0)
        }
    }

    // -------------------------------------------------------------------------
    // Spectral rolloff – real FFT implementation
    // -------------------------------------------------------------------------

    /// Compute spectral rolloff frequency in Hz.
    ///
    /// Finds the frequency bin k where cumulative power ≥ 85 % of total power.
    fn compute_spectral_rolloff(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let (mags, n) = self.compute_windowed_spectrum(samples)?;
        let freq_resolution = self.sample_rate as f64 / n as f64;

        // Power = |X[k]|²
        let power: Vec<f64> = mags.iter().map(|&m| m * m).collect();
        let total_power: f64 = power.iter().sum();

        if total_power < 1e-24 {
            return Ok(0.0);
        }

        let rolloff_threshold = total_power * 0.85;
        let mut cumulative = 0.0f64;

        for (k, &p) in power.iter().enumerate() {
            cumulative += p;
            if cumulative >= rolloff_threshold {
                return Ok((k as f64 * freq_resolution) as f32);
            }
        }

        // Fallback: Nyquist
        Ok((self.sample_rate as f64 / 2.0) as f32)
    }

    // -------------------------------------------------------------------------
    // Spectral bandwidth – real FFT implementation
    // -------------------------------------------------------------------------

    /// Compute spectral bandwidth in Hz.
    ///
    /// sqrt(Σ(|f_k - centroid|² * |X[k]|) / Σ|X[k]|)
    fn compute_spectral_bandwidth(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let (mags, n) = self.compute_windowed_spectrum(samples)?;
        let freq_resolution = self.sample_rate as f64 / n as f64;

        let mut magnitude_sum = 0.0f64;
        let mut weighted_centroid = 0.0f64;

        for (k, &mag) in mags.iter().enumerate() {
            let freq_hz = k as f64 * freq_resolution;
            weighted_centroid += freq_hz * mag;
            magnitude_sum += mag;
        }

        if magnitude_sum < 1e-12 {
            return Ok(0.0);
        }

        let centroid_hz = weighted_centroid / magnitude_sum;

        let mut weighted_variance = 0.0f64;
        for (k, &mag) in mags.iter().enumerate() {
            let freq_hz = k as f64 * freq_resolution;
            let deviation = freq_hz - centroid_hz;
            weighted_variance += deviation * deviation * mag;
        }

        Ok(((weighted_variance / magnitude_sum).sqrt()) as f32)
    }

    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / (samples.len() - 1) as f32
    }

    // -------------------------------------------------------------------------
    // MFCC – real FFT + mel filterbank + DCT-II
    // -------------------------------------------------------------------------

    /// Compute 13 MFCC coefficients using real DSP.
    ///
    /// Pipeline:
    /// 1. Hann-windowed rfft → power spectrum
    /// 2. 26 triangular mel filters (80 Hz – Nyquist), equal mel spacing
    /// 3. log(filterbank_energy + 1e-8)
    /// 4. DCT-II: 13 coefficients
    fn compute_mfcc(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        const NUM_MEL_FILTERS: usize = 26;
        const NUM_MFCC: usize = 13;
        const MIN_FREQ_HZ: f64 = 80.0;

        let (mags, n) = self.compute_windowed_spectrum(samples)?;
        let max_freq_hz = self.sample_rate as f64 / 2.0;

        // Power spectrum
        let power: Vec<f64> = mags.iter().map(|&m| m * m).collect();

        // Hz ↔ Mel conversions
        let hz_to_mel = |hz: f64| -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |mel: f64| -> f64 { 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0) };

        let min_mel = hz_to_mel(MIN_FREQ_HZ);
        let max_mel = hz_to_mel(max_freq_hz);

        // NUM_MEL_FILTERS + 2 equally-spaced mel points
        let mel_points: Vec<f64> = (0..NUM_MEL_FILTERS + 2)
            .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (NUM_MEL_FILTERS + 1) as f64)
            .collect();

        // Convert to FFT bin indices  (half-spectrum has n/2+1 bins)
        let freq_resolution = self.sample_rate as f64 / n as f64;
        let hz_to_bin = |hz: f64| -> usize {
            let bin = (hz / freq_resolution).round() as usize;
            bin.min(power.len() - 1)
        };

        let bins: Vec<usize> = mel_points
            .iter()
            .map(|&m| hz_to_bin(mel_to_hz(m)))
            .collect();

        // Build 26 triangular mel filters and compute filterbank energies
        let mut filterbank_energy = vec![0.0f64; NUM_MEL_FILTERS];

        for filter_idx in 0..NUM_MEL_FILTERS {
            let b0 = bins[filter_idx];
            let b1 = bins[filter_idx + 1];
            let b2 = bins[filter_idx + 2];

            // Rising slope
            if b1 > b0 {
                for k in b0..b1 {
                    let weight = (k - b0) as f64 / (b1 - b0) as f64;
                    if k < power.len() {
                        filterbank_energy[filter_idx] += weight * power[k];
                    }
                }
            }

            // Falling slope
            if b2 > b1 {
                for k in b1..=b2 {
                    let weight = (b2 - k) as f64 / (b2 - b1) as f64;
                    if k < power.len() {
                        filterbank_energy[filter_idx] += weight * power[k];
                    }
                }
            }
        }

        // Log filterbank energies
        let log_mel: Vec<f64> = filterbank_energy.iter().map(|&e| (e + 1e-8).ln()).collect();

        // DCT-II: mfcc[i] = sqrt(2/26) * Σ_j(log_mel[j] * cos(π*i*(j+0.5)/26))  for i=1..=13
        let scale = (2.0 / NUM_MEL_FILTERS as f64).sqrt();
        let mfcc: Vec<f32> = (1..=NUM_MFCC)
            .map(|i| {
                let coeff: f64 = log_mel
                    .iter()
                    .enumerate()
                    .map(|(j, &lm)| {
                        lm * (std::f64::consts::PI * i as f64 * (j as f64 + 0.5)
                            / NUM_MEL_FILTERS as f64)
                            .cos()
                    })
                    .sum();
                (scale * coeff) as f32
            })
            .collect();

        Ok(mfcc)
    }

    // -------------------------------------------------------------------------
    // Formant estimation – LPC-based
    // -------------------------------------------------------------------------

    /// Extract formant frequencies F1, F2, F3 using LPC analysis.
    ///
    /// LPC order = 2 + round(sample_rate / 1000), capped at 50.
    /// Levinson-Durbin recursion over the autocorrelation sequence,
    /// then peak-pick the LPC spectral envelope on 512 frequency points.
    fn extract_formants(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let order = (2 + (self.sample_rate / 1000) as usize).min(50);
        let n = samples.len();

        if n < order + 2 {
            // Not enough samples: return generic defaults
            return Ok(vec![500.0, 1500.0, 2500.0]);
        }

        // Pre-emphasis filter
        let mut emphasized = vec![0.0f64; n];
        emphasized[0] = samples[0] as f64;
        for i in 1..n {
            emphasized[i] = samples[i] as f64 - 0.97 * samples[i - 1] as f64;
        }

        // Autocorrelation r[0..=order]
        let mut r = vec![0.0f64; order + 1];
        for k in 0..=order {
            let sum: f64 = (0..n - k).map(|i| emphasized[i] * emphasized[i + k]).sum();
            r[k] = sum;
        }

        if r[0] < 1e-12 {
            // Silent frame
            return Ok(vec![500.0, 1500.0, 2500.0]);
        }

        // Levinson-Durbin recursion
        // Produces LPC coefficients a[1..=order] with a[0] = 1
        let mut a = vec![0.0f64; order + 1];
        let mut a_prev = vec![0.0f64; order + 1];
        let mut error = r[0];

        for m in 1..=order {
            // Reflection coefficient k_m
            let mut lambda = 0.0f64;
            for j in 1..m {
                lambda += a_prev[j] * r[m - j];
            }
            lambda = -(r[m] + lambda) / error;

            // Update LPC coefficients
            a[m] = lambda;
            for j in 1..m {
                a[j] = a_prev[j] + lambda * a_prev[m - j];
            }

            // Update error
            error *= 1.0 - lambda * lambda;
            if error.abs() < 1e-12 {
                break;
            }

            // Snapshot for next iteration
            a_prev[..=m].copy_from_slice(&a[..=m]);
        }

        // Evaluate LPC spectral envelope |1/A(e^{jω})| on 512 frequencies
        let num_freq = 512usize;
        let mut envelope = vec![0.0f32; num_freq];
        let fs = self.sample_rate as f64;

        for fi in 0..num_freq {
            let omega = std::f64::consts::PI * fi as f64 / (num_freq - 1) as f64;
            // A(e^{jω}) = 1 + a[1]*e^{-jω} + a[2]*e^{-j2ω} + ...
            let mut re = 1.0f64;
            let mut im = 0.0f64;
            for k in 1..=order {
                re += a[k] * (k as f64 * omega).cos();
                im -= a[k] * (k as f64 * omega).sin();
            }
            let magnitude = (re * re + im * im).sqrt();
            envelope[fi] = if magnitude > 1e-12 {
                (1.0 / magnitude) as f32
            } else {
                0.0
            };
        }

        // Peak-pick in F1 [200-1000 Hz], F2 [800-2500 Hz], F3 [2000-4000 Hz]
        let freq_at =
            |bin: usize| -> f32 { (bin as f64 * fs / (2.0 * (num_freq - 1) as f64)) as f32 };
        let bin_of =
            |hz: f64| -> usize { ((hz * (num_freq - 1) as f64 * 2.0) / fs).round() as usize };

        let find_peak = |lo_hz: f64, hi_hz: f64| -> f32 {
            let lo = bin_of(lo_hz).min(num_freq - 1);
            let hi = bin_of(hi_hz).min(num_freq - 1);
            if lo >= hi {
                return (lo_hz + hi_hz) as f32 / 2.0;
            }
            let mut best_bin = lo;
            let mut best_val = envelope[lo];
            for b in lo..=hi {
                if envelope[b] > best_val {
                    best_val = envelope[b];
                    best_bin = b;
                }
            }
            freq_at(best_bin)
        };

        let f1 = find_peak(200.0, 1000.0);
        let f2 = find_peak(800.0, 2500.0);
        let f3 = find_peak(2000.0, 4000.0);

        Ok(vec![f1, f2, f3])
    }

    // -------------------------------------------------------------------------
    // Jitter / Shimmer (unchanged, already use valid DSP)
    // -------------------------------------------------------------------------

    /// Compute jitter (pitch irregularity)
    fn compute_jitter(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let f0_contour = self.extract_f0_contour(samples)?;
        if f0_contour.len() < 2 {
            return Ok(0.0);
        }

        let mut period_diffs = Vec::new();
        for i in 1..f0_contour.len() {
            if f0_contour[i] > 0.0 && f0_contour[i - 1] > 0.0 {
                let period1 = 1.0 / f0_contour[i - 1];
                let period2 = 1.0 / f0_contour[i];
                period_diffs.push((period2 - period1).abs());
            }
        }

        if period_diffs.is_empty() {
            Ok(0.0)
        } else {
            Ok(self.mean(&period_diffs))
        }
    }

    /// Compute shimmer (amplitude irregularity)
    fn compute_shimmer(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        if energy_contour.len() < 2 {
            return Ok(0.0);
        }

        let mut amplitude_diffs = Vec::new();
        for i in 1..energy_contour.len() {
            let diff = (energy_contour[i] - energy_contour[i - 1]).abs();
            amplitude_diffs.push(diff);
        }

        Ok(self.mean(&amplitude_diffs))
    }

    // -------------------------------------------------------------------------
    // HNR – autocorrelation-based
    // -------------------------------------------------------------------------

    /// Compute Harmonic-to-Noise Ratio in dB via normalized autocorrelation.
    ///
    /// Searches pitch-period lags corresponding to 80–500 Hz.
    /// HNR = 10 * log10(r_peak / (1 - r_peak + 1e-10))
    /// Returns 0.0 dB for unvoiced frames (r_peak < 0.1).
    fn compute_hnr(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let n = samples.len();
        if n < 2 {
            return Ok(0.0);
        }

        // r[0] = Σ x[i]²
        let r0: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
        if r0 < 1e-12 {
            return Ok(0.0);
        }

        // Search lags in pitch range 80–500 Hz → lag range [sr/500, sr/80]
        let sr = self.sample_rate as usize;
        let lag_min = (sr / 500).max(1);
        let lag_max = (sr / 80).min(n - 1);

        if lag_min > lag_max {
            return Ok(0.0);
        }

        let mut r_peak = 0.0f64;

        for tau in lag_min..=lag_max {
            let r_tau: f64 = (0..n - tau)
                .map(|i| (samples[i] as f64) * (samples[i + tau] as f64))
                .sum::<f64>()
                / r0;
            if r_tau > r_peak {
                r_peak = r_tau;
            }
        }

        // Unvoiced criterion
        if r_peak < 0.1 {
            return Ok(0.0);
        }

        let hnr_db = 10.0 * (r_peak / (1.0 - r_peak + 1e-10)).log10();
        Ok(hnr_db as f32)
    }

    /// Compute breathiness measure
    fn compute_breathiness(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified breathiness computation based on noise content
        let hnr = self.compute_hnr(samples)?;
        Ok((20.0 - hnr).max(0.0) / 20.0) // Inverse of HNR, normalized
    }

    /// Detect vocal fry
    fn detect_vocal_fry(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified vocal fry detection based on low frequency irregularities
        let f0_contour = self.extract_f0_contour(samples)?;
        let low_f0_ratio = f0_contour
            .iter()
            .filter(|&&f0| f0 > 0.0 && f0 < 80.0) // Very low F0
            .count() as f32
            / f0_contour.len() as f32;

        Ok(low_f0_ratio)
    }

    /// Compute creakiness measure
    fn compute_creakiness(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified creakiness based on amplitude irregularities at low frequencies
        let shimmer = self.compute_shimmer(samples)?;
        let vocal_fry = self.detect_vocal_fry(samples)?;

        Ok((shimmer + vocal_fry) / 2.0)
    }

    /// Compute speech ratio (speech vs silence)
    fn compute_speech_ratio(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        let mean_energy = self.mean(&energy_contour);
        let threshold = mean_energy * 0.1;

        let speech_frames = energy_contour
            .iter()
            .filter(|&&energy| energy > threshold)
            .count();

        Ok(speech_frames as f32 / energy_contour.len() as f32)
    }

    /// Compute articulation rate
    fn compute_articulation_rate(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Articulation rate based on spectral changes (FFT-based centroid)
        let zcr = self.compute_zero_crossing_rate(samples);
        let spectral_centroid = self.compute_spectral_centroid(samples)?;

        // Higher ZCR and spectral centroid suggest clearer articulation
        Ok((zcr + spectral_centroid / 8000.0).min(1.0))
    }

    /// Compute tempo variation
    fn compute_tempo_variation(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        if energy_contour.len() < 3 {
            return Ok(0.0);
        }

        // Compute tempo based on energy peaks
        let mut intervals = Vec::new();
        let threshold = self.mean(&energy_contour) * 0.7;
        let mut last_peak = 0;

        for (i, &energy) in energy_contour.iter().enumerate() {
            if energy > threshold && i > last_peak + 3 {
                // Avoid double peaks
                if last_peak > 0 {
                    intervals.push((i - last_peak) as f32);
                }
                last_peak = i;
            }
        }

        if intervals.is_empty() {
            Ok(0.0)
        } else {
            Ok(self.std_dev(&intervals) / self.mean(&intervals))
        }
    }

    // -------------------------------------------------------------------------
    // Statistical utilities
    // -------------------------------------------------------------------------

    fn mean(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn std_dev(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = self.mean(values);
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;

        variance.sqrt()
    }

    fn variance(&self, values: &[f32]) -> f32 {
        self.std_dev(values).powi(2)
    }

    fn min(&self, values: &[f32]) -> f32 {
        values.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    fn max(&self, values: &[f32]) -> f32 {
        values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    fn linear_trend(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let sum_x = (0..values.len()).sum::<usize>() as f32;
        let sum_y = values.iter().sum::<f32>();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }
}

impl Default for EmotionFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feature_extraction() {
        let extractor = EmotionFeatureExtractor::new();

        // Create test audio
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = AudioBuffer::mono(samples, 16000);

        let features = extractor.extract_emotion_features(&audio).await;
        assert!(features.is_ok());

        let features = features.unwrap();
        assert!(features.contains_key("f0_mean"));
        assert!(features.contains_key("energy_mean"));
        assert!(features.contains_key("spectral_centroid"));
        assert!(features.contains_key("jitter"));
    }

    #[test]
    fn test_prosodic_features() {
        let extractor = EmotionFeatureExtractor::new();
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();

        let features = extractor.extract_prosodic_features(&samples);
        assert!(features.is_ok());

        let features = features.unwrap();
        assert!(features.contains_key("f0_mean"));
        assert!(features.contains_key("energy_mean"));
        assert!(features.contains_key("speaking_rate"));
    }

    #[test]
    fn test_spectral_features() {
        let extractor = EmotionFeatureExtractor::new();
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();

        let features = extractor.extract_spectral_features(&samples);
        assert!(features.is_ok());

        let features = features.unwrap();
        assert!(features.contains_key("spectral_centroid"));
        assert!(features.contains_key("zero_crossing_rate"));
        assert!(features.contains_key("mfcc_0"));
    }

    #[test]
    fn test_statistical_functions() {
        let extractor = EmotionFeatureExtractor::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(extractor.mean(&values), 3.0);
        assert!(extractor.std_dev(&values) > 0.0);
        assert_eq!(extractor.min(&values), 1.0);
        assert_eq!(extractor.max(&values), 5.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let extractor = EmotionFeatureExtractor::new();
        let samples = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = extractor.compute_zero_crossing_rate(&samples);
        assert!(zcr > 0.0);
    }

    // -------------------------------------------------------------------------
    // New tests for real FFT-based implementations
    // -------------------------------------------------------------------------

    /// A pure sine at low frequency should have a lower spectral centroid than
    /// the same tone at a higher frequency.
    #[test]
    fn test_spectral_centroid_ordering() {
        let extractor = EmotionFeatureExtractor::new();
        let sr = extractor.sample_rate as f32;
        let n = 4096usize;

        // 500 Hz sine
        let low: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 500.0 * i as f32 / sr).sin())
            .collect();

        // 3000 Hz sine
        let high: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / sr).sin())
            .collect();

        let centroid_low = extractor.compute_spectral_centroid(&low).unwrap();
        let centroid_high = extractor.compute_spectral_centroid(&high).unwrap();

        assert!(
            centroid_low < centroid_high,
            "low-freq centroid {centroid_low} Hz should be less than high-freq centroid {centroid_high} Hz"
        );
    }

    /// `compute_mfcc` must return exactly 13 coefficients.
    #[test]
    fn test_mfcc_length() {
        let extractor = EmotionFeatureExtractor::new();
        let sr = extractor.sample_rate as f32;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let mfcc = extractor.compute_mfcc(&samples).unwrap();
        assert_eq!(mfcc.len(), 13, "MFCC must produce exactly 13 coefficients");
    }

    /// Different vowel-like signals (different harmonic structure) must produce
    /// different formant estimates, not the old hardcoded `[800, 1200, 2500]`.
    #[test]
    fn test_formants_not_hardcoded() {
        let extractor = EmotionFeatureExtractor::new();
        let sr = extractor.sample_rate as f32;
        let n = 8192usize;

        // Signal A: fundamental 100 Hz + harmonics emphasising F1 ~ 700 Hz (vowel /a/-like)
        let signal_a: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * 100.0 * t).sin()
                    + 0.8 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
            })
            .collect();

        // Signal B: fundamental 100 Hz + harmonics emphasising F1 ~ 300 Hz (vowel /i/-like)
        let signal_b: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * 100.0 * t).sin()
                    + 0.8 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 2300.0 * t).sin()
            })
            .collect();

        let formants_a = extractor.extract_formants(&signal_a).unwrap();
        let formants_b = extractor.extract_formants(&signal_b).unwrap();

        // Verify we get 3 formants
        assert_eq!(formants_a.len(), 3);
        assert_eq!(formants_b.len(), 3);

        // Verify they are NOT both the old hardcoded result
        let old_hardcoded = [800.0f32, 1200.0, 2500.0];
        let a_is_hardcoded = formants_a
            .iter()
            .zip(old_hardcoded.iter())
            .all(|(&fa, &fo)| (fa - fo).abs() < 1.0);
        let b_is_hardcoded = formants_b
            .iter()
            .zip(old_hardcoded.iter())
            .all(|(&fb, &fo)| (fb - fo).abs() < 1.0);

        assert!(
            !a_is_hardcoded || !b_is_hardcoded,
            "At least one signal must produce non-hardcoded formants"
        );

        // Verify signals with different spectral envelopes produce different F1 estimates
        // (not strictly required to be dramatically different, but they should differ)
        assert_ne!(
            formants_a[0].round() as i32,
            formants_b[0].round() as i32,
            "Different vowel-like signals should produce different F1 values: {:?} vs {:?}",
            formants_a,
            formants_b
        );
    }

    /// A pure sine wave should have significantly higher HNR than white noise.
    #[test]
    fn test_hnr_tone_vs_noise() {
        let extractor = EmotionFeatureExtractor::new();
        let sr = extractor.sample_rate as f32;
        let n = 8192usize;

        // Pure 200 Hz sine
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin())
            .collect();

        // White noise – use a simple deterministic LCG to avoid rand dependency
        let noise: Vec<f32> = (0..n)
            .scan(12345u64, |state, _| {
                *state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let val = (*state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0;
                Some(val)
            })
            .collect();

        let hnr_tone = extractor.compute_hnr(&tone).unwrap();
        let hnr_noise = extractor.compute_hnr(&noise).unwrap();

        assert!(
            hnr_tone > hnr_noise,
            "Sine HNR ({hnr_tone:.2} dB) should exceed noise HNR ({hnr_noise:.2} dB)"
        );
        // Tone should produce a meaningful voiced HNR
        assert!(
            hnr_tone > 5.0,
            "Sine HNR should be > 5 dB, got {hnr_tone:.2}"
        );
    }
}
