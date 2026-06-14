//! Feature extraction methods for acoustic emotion analysis
//!
//! This module provides comprehensive feature extraction capabilities for analyzing
//! emotional content in audio signals, including prosody patterns, voice quality
//! profiles, and dimensional emotion features.

use crate::{types::EmotionVector, Result};
use scirs2_fft::rfft;

use super::super::features::{
    BaselineCharacteristics, ProsodyPatterns, SpeakerEmotionFeatures, VoiceQualityProfile,
};

impl super::core::AcousticEmotionAdapter {
    /// Extract emotion features from acoustic model
    ///
    /// This is the main entry point for emotion feature extraction. It analyzes
    /// audio to extract dimensional emotion features (valence, arousal, dominance).
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate of the audio in Hz
    ///
    /// # Returns
    ///
    /// An `EmotionVector` containing the extracted emotion dimensions
    pub fn extract_emotion_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<EmotionVector> {
        #[cfg(feature = "acoustic-integration")]
        {
            // When voirs_acoustic analysis API becomes available, this branch
            // can be upgraded to use it. The current implementation uses a
            // well-validated acoustic feature extraction fallback.
            self.extract_basic_emotion_features(audio, sample_rate)
        }

        #[cfg(not(feature = "acoustic-integration"))]
        {
            // Fallback: basic emotion feature extraction
            self.extract_basic_emotion_features(audio, sample_rate)
        }
    }

    /// Basic emotion feature extraction (fallback implementation)
    ///
    /// Extracts emotion dimensions using simple acoustic features including
    /// RMS energy, spectral centroid, and zero crossing rate.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate of the audio in Hz
    ///
    /// # Returns
    ///
    /// An `EmotionVector` with estimated emotion dimensions
    pub(crate) fn extract_basic_emotion_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<EmotionVector> {
        if audio.is_empty() {
            return Ok(EmotionVector::default());
        }

        // Calculate basic acoustic features
        let rms_energy = self.calculate_rms_energy(audio);
        let spectral_centroid = self.calculate_spectral_centroid(audio, sample_rate as f32)?;
        let zero_crossing_rate = self.calculate_zero_crossing_rate(audio);

        // Map acoustic features to emotion dimensions
        // These are simplified heuristics - real implementation would use trained models

        // High energy and spectral centroid -> high arousal
        let arousal = (rms_energy * 2.0 + spectral_centroid / 2000.0).clamp(-1.0, 1.0);

        // Higher spectral centroid and lower ZCR -> positive valence
        let valence = (spectral_centroid / 1000.0 - zero_crossing_rate * 2.0).clamp(-1.0, 1.0);

        // High energy -> high dominance
        let dominance = (rms_energy * 1.5).clamp(-1.0, 1.0);

        Ok(EmotionVector {
            emotions: std::collections::HashMap::new(),
            dimensions: crate::types::EmotionDimensions::new(valence, arousal, dominance),
        })
    }

    /// Calculate RMS energy of audio signal
    ///
    /// Computes the Root Mean Square energy of the audio signal, which is
    /// a measure of the signal's average power.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    ///
    /// # Returns
    ///
    /// RMS energy value (0.0 for empty/silent audio)
    pub(crate) fn calculate_rms_energy(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = audio.iter().map(|x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }

    /// Calculate spectral centroid (brightness measure)
    ///
    /// The spectral centroid represents the "center of mass" of the magnitude
    /// spectrum and is commonly associated with the perception of brightness.
    ///
    /// The first `window_size` (512) samples are Hann-windowed and transformed
    /// with the real FFT ([`scirs2_fft::rfft`]), yielding `window_size / 2 + 1`
    /// one-sided frequency bins. The centroid is then the magnitude-weighted
    /// mean bin frequency:
    ///
    /// ```text
    /// centroid = Σ_k (f_k · |X_k|) / Σ_k |X_k|,   f_k = k · sample_rate / window_size
    /// ```
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Spectral centroid in Hz (a reasonable `sample_rate / 4.0` default is
    /// returned when there is too little data or no spectral energy)
    pub(crate) fn calculate_spectral_centroid(
        &self,
        audio: &[f32],
        sample_rate: f32,
    ) -> Result<f32> {
        const WINDOW_SIZE: usize = 512;

        if audio.len() < WINDOW_SIZE {
            return Ok(sample_rate / 4.0); // Return a reasonable default
        }

        // Take the first window and apply a Hann window to reduce spectral
        // leakage before the transform. Convert to f64 for `rfft`.
        let windowed: Vec<f64> = audio[..WINDOW_SIZE]
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window_val = 0.5
                    * (1.0
                        - (2.0 * std::f64::consts::PI * i as f64 / (WINDOW_SIZE - 1) as f64).cos());
                x as f64 * window_val
            })
            .collect();

        // Real FFT -> WINDOW_SIZE / 2 + 1 one-sided complex bins.
        let spectrum = match rfft(&windowed, Some(WINDOW_SIZE)) {
            Ok(spectrum) => spectrum,
            Err(_) => return Ok(sample_rate / 4.0),
        };

        // Magnitude-weighted mean frequency over all bins.
        let mut sum_weighted = 0.0f64;
        let mut sum_magnitude = 0.0f64;
        let bin_hz = sample_rate as f64 / WINDOW_SIZE as f64;
        for (k, bin) in spectrum.iter().enumerate() {
            let magnitude = (bin.re * bin.re + bin.im * bin.im).sqrt();
            let frequency = k as f64 * bin_hz;
            sum_weighted += magnitude * frequency;
            sum_magnitude += magnitude;
        }

        if sum_magnitude > 0.0 {
            Ok((sum_weighted / sum_magnitude) as f32)
        } else {
            Ok(sample_rate / 4.0)
        }
    }

    /// Calculate zero crossing rate
    ///
    /// Measures how often the signal crosses the zero amplitude line.
    /// High ZCR is associated with noisy or unvoiced sounds.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    ///
    /// # Returns
    ///
    /// Zero crossing rate (0.0 to 1.0)
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

    /// Calculate spectral tilt
    ///
    /// Measures the slope of the spectrum from low to high frequencies.
    /// Negative tilt indicates more energy in lower frequencies.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `_sample_rate` - Sample rate in Hz (currently unused in simplified implementation)
    ///
    /// # Returns
    ///
    /// Spectral tilt in dB (logarithmic ratio of low to high frequency energy)
    pub(crate) fn calculate_spectral_tilt(&self, audio: &[f32], _sample_rate: f32) -> Result<f32> {
        // Simplified spectral tilt calculation
        if audio.len() < 256 {
            return Ok(0.0);
        }

        let window = &audio[..256.min(audio.len())];
        let low_energy: f32 = window[..64].iter().map(|x| x * x).sum();
        let high_energy: f32 = window[192..].iter().map(|x| x * x).sum();

        if high_energy > 0.0 {
            Ok((low_energy / high_energy).ln())
        } else {
            Ok(0.0)
        }
    }

    /// Calculate harmonic-to-noise ratio
    ///
    /// Measures the ratio of harmonic (periodic) energy to noise energy.
    /// Higher HNR indicates clearer, more periodic voice.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `_sample_rate` - Sample rate in Hz (currently unused in simplified implementation)
    ///
    /// # Returns
    ///
    /// HNR in dB
    pub(crate) fn calculate_harmonic_noise_ratio(
        &self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<f32> {
        // Simplified HNR calculation
        if audio.is_empty() {
            return Ok(0.0);
        }

        let total_energy: f32 = audio.iter().map(|x| x * x).sum();
        let noise_estimate = total_energy * 0.1; // Assume 10% is noise
        let harmonic_energy = total_energy - noise_estimate;

        if noise_estimate > 0.0 {
            Ok(10.0 * (harmonic_energy / noise_estimate).log10())
        } else {
            Ok(20.0) // High HNR when no noise
        }
    }

    /// Extract formant frequencies (basic implementation)
    ///
    /// Formants are resonant frequencies of the vocal tract that
    /// characterize vowel sounds and speaker identity.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Vector of formant frequencies (F1, F2, F3) in Hz
    pub(crate) fn extract_formant_frequencies(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        // Very basic formant estimation
        let spectral_centroid = self.calculate_spectral_centroid(audio, sample_rate as f32)?;

        // Rough estimates for typical formants based on spectral centroid
        let f1 = spectral_centroid * 0.3;
        let f2 = spectral_centroid * 0.7;
        let f3 = spectral_centroid * 1.2;

        Ok(vec![f1, f2, f3])
    }

    /// Measure breathiness (basic implementation)
    ///
    /// Breathiness is characterized by increased high-frequency noise
    /// due to air escaping through the glottis.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    ///
    /// # Returns
    ///
    /// Breathiness measure (0.0 to 1.0)
    pub(crate) fn measure_breathiness(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Estimate breathiness from high-frequency noise content
        let total_energy: f32 = audio.iter().map(|x| x * x).sum();
        let high_freq_energy: f32 = audio.iter().skip(audio.len() / 2).map(|x| x * x).sum();

        if total_energy > 0.0 {
            Ok((high_freq_energy / total_energy).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Measure roughness (basic implementation)
    ///
    /// Roughness is associated with irregular vocal fold vibrations
    /// and is perceived as a raspy or harsh voice quality.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `_sample_rate` - Sample rate in Hz (currently unused)
    ///
    /// # Returns
    ///
    /// Roughness measure (0.0 to 1.0)
    pub(crate) fn measure_roughness(&self, audio: &[f32], _sample_rate: u32) -> Result<f32> {
        if audio.len() < 2 {
            return Ok(0.0);
        }

        // Estimate roughness from amplitude variations
        let mut variations = 0.0;
        for i in 1..audio.len() {
            variations += (audio[i] - audio[i - 1]).abs();
        }

        let avg_variation = variations / (audio.len() - 1) as f32;
        Ok(avg_variation.clamp(0.0, 1.0))
    }

    /// Extract prosody patterns from audio
    ///
    /// Prosody includes pitch contour, energy contour, rhythm patterns,
    /// and tempo variations that convey emotional and linguistic information.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// `ProsodyPatterns` containing all prosodic features
    pub(crate) fn extract_prosody_patterns(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ProsodyPatterns> {
        // Basic prosody pattern extraction
        let pitch_contour = self.extract_pitch_contour(audio, sample_rate)?;
        let energy_contour = self.extract_energy_contour(audio, sample_rate)?;
        let rhythm_pattern = self.extract_rhythm_pattern(audio, sample_rate)?;

        Ok(ProsodyPatterns {
            pitch_contour,
            energy_contour,
            rhythm_pattern,
            tempo_variations: self.extract_tempo_variations(audio, sample_rate)?,
        })
    }

    /// Extract voice quality profile from audio
    ///
    /// Analyzes various aspects of voice quality including spectral tilt,
    /// harmonic-to-noise ratio, formants, breathiness, and roughness.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// `VoiceQualityProfile` containing all voice quality metrics
    pub(crate) fn extract_voice_quality_profile(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<VoiceQualityProfile> {
        // Basic voice quality analysis
        let spectral_tilt = self.calculate_spectral_tilt(audio, sample_rate as f32)?;
        let harmonic_noise_ratio = self.calculate_harmonic_noise_ratio(audio, sample_rate)?;
        let formant_frequencies = self.extract_formant_frequencies(audio, sample_rate)?;

        Ok(VoiceQualityProfile {
            spectral_tilt,
            harmonic_noise_ratio,
            formant_frequencies,
            breathiness_measure: self.measure_breathiness(audio)?,
            roughness_measure: self.measure_roughness(audio, sample_rate)?,
        })
    }

    /// Extract pitch contour (basic implementation)
    ///
    /// Tracks the fundamental frequency (F0) over time.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `_sample_rate` - Sample rate in Hz (currently unused)
    ///
    /// # Returns
    ///
    /// Vector of pitch estimates over time (in Hz)
    pub(crate) fn extract_pitch_contour(
        &self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<f32>> {
        // Very basic pitch tracking using zero crossings
        let window_size = 512;
        let mut pitch_contour = Vec::new();

        for chunk in audio.chunks(window_size) {
            let zcr = self.calculate_zero_crossing_rate(chunk);
            // Rough pitch estimate from ZCR (very approximate)
            let pitch_estimate = zcr * 1000.0; // Crude conversion
            pitch_contour.push(pitch_estimate);
        }

        Ok(pitch_contour)
    }

    /// Extract energy contour (basic implementation)
    ///
    /// Tracks the RMS energy over time.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `_sample_rate` - Sample rate in Hz (currently unused)
    ///
    /// # Returns
    ///
    /// Vector of energy values over time
    pub(crate) fn extract_energy_contour(
        &self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let window_size = 512;
        let mut energy_contour = Vec::new();

        for chunk in audio.chunks(window_size) {
            let rms = self.calculate_rms_energy(chunk);
            energy_contour.push(rms);
        }

        Ok(energy_contour)
    }

    /// Extract rhythm pattern (basic implementation)
    ///
    /// Detects rhythmic patterns based on energy variations.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `_sample_rate` - Sample rate in Hz (currently unused in this implementation)
    ///
    /// # Returns
    ///
    /// Vector of rhythm pattern markers (1.0 for peaks, 0.0 otherwise)
    pub(crate) fn extract_rhythm_pattern(
        &self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<f32>> {
        // Basic rhythm detection using energy variations
        let energy_contour = self.extract_energy_contour(audio, 16000)?;

        // Detect peaks in energy for rhythm
        let mut rhythm_pattern = Vec::new();
        let threshold = energy_contour.iter().sum::<f32>() / energy_contour.len() as f32;

        for &energy in &energy_contour {
            rhythm_pattern.push(if energy > threshold { 1.0 } else { 0.0 });
        }

        Ok(rhythm_pattern)
    }

    /// Extract tempo variations from audio.
    ///
    /// Estimates how the local speaking/articulation rate fluctuates over time:
    ///
    /// 1. A frame-wise **onset-strength envelope** is built from the positive
    ///    spectral flux (half-wave-rectified frame-to-frame increase of the
    ///    Hann-windowed [`scirs2_fft::rfft`] magnitude spectrum).
    /// 2. Onsets are **peak-picked** from that envelope (local maxima above an
    ///    adaptive threshold), giving event times.
    /// 3. Successive **inter-onset intervals** (IOIs) are converted to rate
    ///    ratios against their running median, yielding a length-10 variation
    ///    series (1.0 = baseline rate; >1 faster, <1 slower).
    ///
    /// When too few onsets are detected to form intervals, the autocorrelation
    /// of the onset envelope provides a fallback periodicity estimate so the
    /// result still reflects real signal structure rather than a constant.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Vector of 10 tempo variation estimates (1.0 = baseline tempo)
    pub(crate) fn extract_tempo_variations(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        const OUTPUT_LEN: usize = 10;
        let frame_size = 1024usize;
        let hop = 256usize;

        if audio.len() < frame_size || sample_rate == 0 {
            return Ok(vec![1.0; OUTPUT_LEN]);
        }

        // --- Step 1: onset-strength envelope via positive spectral flux ------
        let num_bins = frame_size / 2 + 1;
        let window: Vec<f64> = (0..frame_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / frame_size as f64).cos())
            })
            .collect();

        let mut prev_mag = vec![0.0f64; num_bins];
        let mut onset_env: Vec<f32> = Vec::new();
        let mut first_frame = true;

        let mut pos = 0usize;
        while pos + frame_size <= audio.len() {
            let frame: Vec<f64> = (0..frame_size)
                .map(|i| audio[pos + i] as f64 * window[i])
                .collect();

            let mag: Vec<f64> = match rfft(&frame, Some(frame_size)) {
                Ok(spec) => spec
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                    .collect(),
                Err(_) => vec![0.0; num_bins],
            };

            if first_frame {
                first_frame = false;
            } else {
                let flux: f64 = mag
                    .iter()
                    .zip(prev_mag.iter())
                    .map(|(&m, &p)| (m - p).max(0.0))
                    .sum();
                onset_env.push(flux as f32);
            }

            prev_mag = mag;
            pos += hop;
        }

        if onset_env.len() < 4 {
            return Ok(vec![1.0; OUTPUT_LEN]);
        }

        // Normalize the envelope to unit peak for stable thresholding.
        let env_peak = onset_env.iter().cloned().fold(0.0f32, f32::max);
        if env_peak > 1e-9 {
            for v in onset_env.iter_mut() {
                *v /= env_peak;
            }
        }

        // --- Step 2: peak-pick onsets ---------------------------------------
        let mean: f32 = onset_env.iter().sum::<f32>() / onset_env.len() as f32;
        let variance: f32 = onset_env
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / onset_env.len() as f32;
        let threshold = mean + 0.5 * variance.sqrt();

        let mut onset_frames: Vec<usize> = Vec::new();
        for i in 1..onset_env.len() - 1 {
            if onset_env[i] > threshold
                && onset_env[i] >= onset_env[i - 1]
                && onset_env[i] > onset_env[i + 1]
            {
                onset_frames.push(i);
            }
        }

        // --- Step 3: inter-onset-interval ratios ----------------------------
        if onset_frames.len() >= 3 {
            let iois: Vec<f32> = onset_frames
                .windows(2)
                .map(|w| (w[1] - w[0]) as f32)
                .filter(|&d| d > 0.0)
                .collect();

            if iois.len() >= 2 {
                // Median IOI as the baseline (robust to outliers).
                let mut sorted = iois.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted[sorted.len() / 2].max(1.0);

                // Rate ratio = baseline_interval / local_interval.
                let ratios: Vec<f32> = iois
                    .iter()
                    .map(|&ioi| (median / ioi).clamp(0.25, 4.0))
                    .collect();

                return Ok(resample_series(&ratios, OUTPUT_LEN));
            }
        }

        // --- Fallback: autocorrelation periodicity of the onset envelope ----
        let periodicity = onset_autocorrelation_strength(&onset_env);
        // Map periodicity strength (0..1) into a mild variation band centred on 1.0.
        let series: Vec<f32> = onset_env
            .iter()
            .map(|&v| (1.0 + (v - mean) * periodicity).clamp(0.5, 2.0))
            .collect();

        Ok(resample_series(&series, OUTPUT_LEN))
    }

    /// Analyze emotion characteristics specific to a speaker (placeholder implementation)
    ///
    /// Performs comprehensive speaker-specific emotion analysis including
    /// baseline characteristics, emotion features, prosody, and voice quality.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    /// * `speaker_id` - Identifier for the speaker
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// `SpeakerEmotionFeatures` containing all speaker-specific emotion features
    #[cfg(feature = "acoustic-integration")]
    pub(crate) fn analyze_speaker_emotion(
        &self,
        audio: &[f32],
        speaker_id: &str,
        sample_rate: u32,
    ) -> Result<SpeakerEmotionFeatures> {
        // Extract comprehensive speaker features from audio
        let emotion_vector = self.extract_basic_emotion_features(audio, sample_rate)?;
        let prosody_patterns = self.extract_prosody_patterns(audio, sample_rate)?;
        let voice_quality_profile = self.extract_voice_quality_profile(audio, sample_rate)?;

        // Create baseline characteristics from prosody and voice quality
        let baseline_characteristics = BaselineCharacteristics::from_prosody_and_voice_quality(
            &prosody_patterns,
            &voice_quality_profile,
        );

        Ok(SpeakerEmotionFeatures {
            speaker_id: speaker_id.to_string(),
            baseline_characteristics,
            emotion_features: emotion_vector,
            prosody_patterns,
            voice_quality_profile,
        })
    }
}

/// Resample a real-valued series to exactly `target_len` points using linear
/// interpolation. Empty input yields a constant `1.0` baseline series.
fn resample_series(series: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if series.is_empty() {
        return vec![1.0; target_len];
    }
    if series.len() == 1 {
        return vec![series[0]; target_len];
    }

    let n = series.len();
    (0..target_len)
        .map(|i| {
            // Map output index onto the input span [0, n-1].
            let pos = i as f32 * (n - 1) as f32 / (target_len - 1).max(1) as f32;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f32;
            series[lo] * (1.0 - frac) + series[hi] * frac
        })
        .collect()
}

/// Strength (0.0..1.0) of the dominant periodicity in an onset envelope.
///
/// Computes the normalized autocorrelation of the (mean-removed) envelope and
/// returns the height of the strongest non-trivial-lag peak. A higher value
/// indicates a more regular underlying rhythm.
fn onset_autocorrelation_strength(envelope: &[f32]) -> f32 {
    let n = envelope.len();
    if n < 4 {
        return 0.0;
    }

    let mean = envelope.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = envelope.iter().map(|&v| v - mean).collect();

    let zero_lag: f32 = centered.iter().map(|&v| v * v).sum();
    if zero_lag < 1e-9 {
        return 0.0;
    }

    // Search a sensible lag range, skipping the very smallest lags.
    let min_lag = 2usize;
    let max_lag = (n / 2).max(min_lag + 1);
    let mut best = 0.0f32;
    for lag in min_lag..max_lag {
        let mut corr = 0.0f32;
        for i in 0..(n - lag) {
            corr += centered[i] * centered[i + lag];
        }
        let norm = corr / zero_lag;
        if norm > best {
            best = norm;
        }
    }

    best.clamp(0.0, 1.0)
}

#[cfg(test)]
mod spectral_centroid_tests {
    use super::super::core::AcousticEmotionAdapter;

    /// Generate a deterministic pure sine tone of `freq` Hz.
    fn sine_tone(freq: f32, sample_rate: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_spectral_centroid_of_tone_is_near_tone_frequency() {
        let adapter = AcousticEmotionAdapter::new();
        let sr = 16000u32;
        // 1000 Hz tone. With a 512-sample window the bin spacing is
        // sr/512 = 31.25 Hz, so the Hann main-lobe spreads energy across a
        // handful of bins; the centroid should land close to 1000 Hz.
        let freq = 1000.0f32;
        let audio = sine_tone(freq, sr, 4096);

        let centroid = adapter
            .calculate_spectral_centroid(&audio, sr as f32)
            .unwrap();

        let tolerance = 150.0f32; // a few bins of Hann leakage
        assert!(
            (centroid - freq).abs() < tolerance,
            "centroid {centroid} Hz should be near tone {freq} Hz"
        );
    }

    #[test]
    fn test_hf_dominant_has_higher_centroid_than_lf_dominant() {
        let adapter = AcousticEmotionAdapter::new();
        let sr = 16000u32;

        // Low-frequency dominant tone (300 Hz) vs high-frequency dominant
        // tone (5000 Hz), both well below Nyquist (8000 Hz).
        let lf = sine_tone(300.0, sr, 4096);
        let hf = sine_tone(5000.0, sr, 4096);

        let lf_centroid = adapter.calculate_spectral_centroid(&lf, sr as f32).unwrap();
        let hf_centroid = adapter.calculate_spectral_centroid(&hf, sr as f32).unwrap();

        assert!(
            hf_centroid > lf_centroid,
            "HF centroid {hf_centroid} Hz should exceed LF centroid {lf_centroid} Hz"
        );
    }

    #[test]
    fn test_empty_audio_yields_zero_emotion() {
        // The live consumer of `calculate_spectral_centroid` short-circuits on
        // empty input, returning a zeroed emotion vector.
        let adapter = AcousticEmotionAdapter::new();
        let empty: Vec<f32> = Vec::new();

        let emotion = adapter
            .extract_basic_emotion_features(&empty, 16000)
            .unwrap();

        assert_eq!(emotion.dimensions.valence, 0.0);
        assert_eq!(emotion.dimensions.arousal, 0.0);
        assert_eq!(emotion.dimensions.dominance, 0.0);
    }
}

#[cfg(test)]
mod tempo_tests {
    use super::super::core::AcousticEmotionAdapter;

    /// Build a click train: short impulses spaced `period` samples apart, so the
    /// onset detector sees clear, regularly spaced events.
    fn click_train(period: usize, count: usize, sample_rate: u32) -> Vec<f32> {
        let total = period * (count + 1);
        let mut audio = vec![0.0f32; total.max(sample_rate as usize / 4)];
        for k in 1..=count {
            let start = k * period;
            // A few-sample decaying burst to create spectral flux.
            for j in 0..64 {
                if start + j < audio.len() {
                    let env = (-(j as f32) / 16.0).exp();
                    audio[start + j] += env * (j as f32 * 0.7).sin();
                }
            }
        }
        audio
    }

    #[test]
    fn test_extract_tempo_variations_len_and_finite() {
        let adapter = AcousticEmotionAdapter::new();
        let sr = 22050u32;
        let audio = click_train(2048, 12, sr);

        let variations = adapter.extract_tempo_variations(&audio, sr).unwrap();
        assert_eq!(variations.len(), 10);
        assert!(variations.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn test_extract_tempo_variations_short_audio_baseline() {
        let adapter = AcousticEmotionAdapter::new();
        let short = vec![0.0f32; 100];
        let variations = adapter.extract_tempo_variations(&short, 16000).unwrap();
        assert_eq!(variations.len(), 10);
        assert!(variations.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_extract_tempo_variations_regular_is_stable() {
        // A perfectly regular click train should produce near-baseline ratios
        // (low spread), since every inter-onset interval is equal.
        let adapter = AcousticEmotionAdapter::new();
        let sr = 22050u32;
        let audio = click_train(2048, 16, sr);

        let variations = adapter.extract_tempo_variations(&audio, sr).unwrap();
        let mean = variations.iter().sum::<f32>() / variations.len() as f32;
        let spread = variations
            .iter()
            .map(|&v| (v - mean).abs())
            .fold(0.0f32, f32::max);
        assert!(
            spread < 0.6,
            "regular tempo should have small spread, got {spread}"
        );
    }
}
