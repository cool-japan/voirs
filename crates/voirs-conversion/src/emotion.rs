//! Emotion control integration for voice conversion
//!
//! This module provides integration with the voirs-emotion crate to enable
//! emotional transformation during voice conversion processes.

#[cfg(feature = "emotion-integration")]
use voirs_emotion;

#[cfg(feature = "emotion-integration")]
use crate::transforms::{PitchTransform, SpeedTransform, Transform};

use crate::{Error, Result};

/// Emotion integration adapter for voice conversion
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionConversionAdapter {
    /// Emotion model configuration
    config: Option<voirs_emotion::config::EmotionConfig>,
    /// Currently active emotion state
    current_emotion: Option<voirs_emotion::types::EmotionState>,
    /// Emotion transfer configuration
    transfer_config: EmotionTransferConfig,
}

#[cfg(feature = "emotion-integration")]
impl EmotionConversionAdapter {
    /// Create new emotion adapter
    pub fn new() -> Self {
        Self {
            config: None,
            current_emotion: None,
            transfer_config: EmotionTransferConfig::default(),
        }
    }

    /// Create adapter with specific emotion configuration
    pub fn with_config(config: voirs_emotion::config::EmotionConfig) -> Self {
        Self {
            config: Some(config),
            current_emotion: None,
            transfer_config: EmotionTransferConfig::default(),
        }
    }

    /// Create adapter with emotion transfer configuration
    pub fn with_transfer_config(
        emotion_config: voirs_emotion::config::EmotionConfig,
        transfer_config: EmotionTransferConfig,
    ) -> Self {
        Self {
            config: Some(emotion_config),
            current_emotion: None,
            transfer_config,
        }
    }

    /// Set target emotion for conversion
    pub fn set_target_emotion(&mut self, emotion: voirs_emotion::types::EmotionState) {
        self.current_emotion = Some(emotion);
    }

    /// Convert voice with emotional transformation
    pub async fn convert_with_emotion(
        &self,
        input_audio: &[f32],
        target_emotion: &voirs_emotion::types::EmotionState,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        if intensity < 0.0 || intensity > 1.0 {
            return Err(Error::validation(
                "Emotion intensity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if input_audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        // Extract emotion parameters for conversion
        let emotion_params = self.extract_emotion_parameters(target_emotion, intensity);

        // Apply comprehensive emotional transformation
        let mut output = input_audio.to_vec();

        // Apply pitch modification based on arousal
        if emotion_params.pitch_modification.abs() > 0.01 {
            self.apply_pitch_modulation(&mut output, emotion_params.pitch_modification)?;
        }

        // Apply formant shifting based on valence
        if emotion_params.formant_shift.abs() > 0.01 {
            self.apply_formant_modification(&mut output, emotion_params.formant_shift)?;
        }

        // Apply rhythm adjustment based on dominance
        if emotion_params.rhythm_adjustment.abs() > 0.01 {
            self.apply_rhythm_modification(&mut output, emotion_params.rhythm_adjustment)?;
        }

        // Apply spectral tilt for emotional coloring
        if emotion_params.spectral_tilt.abs() > 0.01 {
            self.apply_spectral_tilt(&mut output, emotion_params.spectral_tilt)?;
        }

        // Apply overall emotional intensity scaling
        self.apply_intensity_scaling(&mut output, intensity)?;

        Ok(output)
    }

    /// Convert voice preserving source emotion while changing speaker
    pub async fn convert_preserving_emotion(
        &self,
        input_audio: &[f32],
        target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        // Extract emotion from source
        let source_emotion = self.detect_source_emotion(input_audio)?;

        // Apply speaker conversion while preserving emotional state
        let converted_audio = self.apply_speaker_conversion_with_emotion_preservation(
            input_audio,
            target_characteristics,
            &source_emotion,
        )?;

        Ok(converted_audio)
    }

    /// Detect emotional state from audio
    pub fn detect_source_emotion(
        &self,
        audio: &[f32],
    ) -> Result<voirs_emotion::types::EmotionState> {
        if audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        // Analyze audio characteristics to detect emotion
        let emotion_features = self.extract_emotion_features(audio)?;

        // Convert features to emotion state
        let emotion_state = self.features_to_emotion_state(emotion_features)?;

        Ok(emotion_state)
    }

    /// Transfer emotion from source to target voice while preserving target speaker identity
    pub async fn transfer_emotion_between_speakers(
        &self,
        source_audio: &[f32],
        target_audio: &[f32],
        transfer_intensity: f32,
    ) -> Result<Vec<f32>> {
        if source_audio.is_empty() || target_audio.is_empty() {
            return Err(Error::audio(
                "Both source and target audio must be non-empty".to_string(),
            ));
        }

        if transfer_intensity < 0.0 || transfer_intensity > 1.0 {
            return Err(Error::validation(
                "Transfer intensity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Detect emotion from source
        let source_emotion = self.detect_source_emotion(source_audio)?;

        // Detect baseline emotion from target
        let target_emotion = self.detect_source_emotion(target_audio)?;

        // Blend emotions based on transfer intensity
        let blended_emotion =
            self.blend_emotions(&source_emotion, &target_emotion, transfer_intensity)?;

        // Apply blended emotion to target audio
        self.convert_with_emotion(target_audio, &blended_emotion, transfer_intensity)
            .await
    }

    /// Blend two emotional states with specified mixing ratio
    pub fn blend_emotions(
        &self,
        emotion_a: &voirs_emotion::types::EmotionState,
        emotion_b: &voirs_emotion::types::EmotionState,
        blend_ratio: f32, // 0.0 = all A, 1.0 = all B
    ) -> Result<voirs_emotion::types::EmotionState> {
        if blend_ratio < 0.0 || blend_ratio > 1.0 {
            return Err(Error::validation(
                "Blend ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Create blended emotion state
        let mut blended_state = emotion_a.clone();

        // Blend dimensional values
        let a_dims = &emotion_a.current.emotion_vector.dimensions;
        let b_dims = &emotion_b.current.emotion_vector.dimensions;

        blended_state.current.emotion_vector.dimensions.arousal =
            a_dims.arousal * (1.0 - blend_ratio) + b_dims.arousal * blend_ratio;
        blended_state.current.emotion_vector.dimensions.valence =
            a_dims.valence * (1.0 - blend_ratio) + b_dims.valence * blend_ratio;
        blended_state.current.emotion_vector.dimensions.dominance =
            a_dims.dominance * (1.0 - blend_ratio) + b_dims.dominance * blend_ratio;

        // Update intensity based on blend
        // Calculate intensity from emotion vector dimensions
        let intensity_a =
            (a_dims.arousal.powi(2) + a_dims.valence.powi(2) + a_dims.dominance.powi(2)).sqrt();
        let intensity_b =
            (b_dims.arousal.powi(2) + b_dims.valence.powi(2) + b_dims.dominance.powi(2)).sqrt();
        let _blended_intensity = intensity_a * (1.0 - blend_ratio) + intensity_b * blend_ratio;

        Ok(blended_state)
    }

    /// Apply gradual emotion transition over time
    pub async fn apply_emotion_transition(
        &self,
        input_audio: &[f32],
        start_emotion: &voirs_emotion::types::EmotionState,
        end_emotion: &voirs_emotion::types::EmotionState,
        transition_points: usize,
    ) -> Result<Vec<f32>> {
        if input_audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        if transition_points == 0 {
            return Err(Error::validation(
                "Transition points must be greater than 0".to_string(),
            ));
        }

        let chunk_size = input_audio.len() / transition_points;
        if chunk_size == 0 {
            return Err(Error::audio(
                "Audio too short for specified transition points".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(input_audio.len());

        for i in 0..transition_points {
            let start_idx = i * chunk_size;
            let end_idx = if i == transition_points - 1 {
                input_audio.len()
            } else {
                (i + 1) * chunk_size
            };

            let chunk = &input_audio[start_idx..end_idx];
            let blend_ratio = i as f32 / (transition_points - 1) as f32;

            // Blend emotions for this chunk
            let current_emotion = self.blend_emotions(start_emotion, end_emotion, blend_ratio)?;

            // Apply emotion to chunk
            let processed_chunk = self
                .convert_with_emotion(chunk, &current_emotion, 0.8)
                .await?;
            output.extend(processed_chunk);
        }

        Ok(output)
    }

    // Private helper methods
    fn extract_emotion_parameters(
        &self,
        emotion: &voirs_emotion::types::EmotionState,
        intensity: f32,
    ) -> EmotionParameters {
        // Extract parameters from the actual EmotionState structure
        let arousal = emotion.current.emotion_vector.dimensions.arousal;
        let valence = emotion.current.emotion_vector.dimensions.valence;
        let dominance = emotion.current.emotion_vector.dimensions.dominance;

        // Scale parameters based on emotion intensity and user-specified intensity
        let emotion_intensity = (arousal.powi(2) + valence.powi(2) + dominance.powi(2)).sqrt();
        let effective_intensity = intensity * emotion_intensity;

        EmotionParameters {
            pitch_modification: arousal * effective_intensity * 0.25,
            formant_shift: valence * effective_intensity * 0.15,
            rhythm_adjustment: dominance * effective_intensity * 0.20,
            spectral_tilt: (arousal - 0.5) * effective_intensity * 0.35,
        }
    }

    fn extract_emotion_features(&self, audio: &[f32]) -> Result<EmotionFeatures> {
        // Analyze audio for emotional characteristics
        let mean_energy = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let peak_energy = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let energy_variance = {
            let mean_square = mean_energy;
            let variance_sum: f32 = audio.iter().map(|&x| (x * x - mean_square).powi(2)).sum();
            variance_sum / audio.len() as f32
        };

        // Estimate pitch variation (simplified)
        let pitch_variation = self.estimate_pitch_variation(audio)?;

        // Estimate spectral characteristics
        let spectral_centroid = self.estimate_spectral_centroid(audio)?;
        let spectral_rolloff = self.estimate_spectral_rolloff(audio)?;

        Ok(EmotionFeatures {
            energy_level: mean_energy.sqrt(),
            energy_variance,
            pitch_variation,
            spectral_centroid,
            spectral_rolloff,
            peak_energy,
        })
    }

    fn estimate_pitch_variation(&self, audio: &[f32]) -> Result<f32> {
        // Real autocorrelation-based F0 estimation.
        // Parameters assume a 22050 Hz sample rate.
        const SAMPLE_RATE: f64 = 22050.0;
        const FRAME_LEN: usize = 551; // ~25 ms at 22050 Hz
        const HOP_LEN: usize = 220; // ~10 ms at 22050 Hz
        const LAG_MIN: usize = 27; // corresponds to ~800 Hz upper bound (sr/800 rounds to 27)
        const LAG_MAX: usize = 275; // corresponds to ~80 Hz lower bound
        const VOICED_THRESHOLD: f64 = 0.3;

        if audio.len() < FRAME_LEN {
            return Ok(0.0);
        }

        let mut voiced_log_f0: Vec<f64> = Vec::new();
        let n_frames = (audio.len() - FRAME_LEN) / HOP_LEN + 1;

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LEN;
            let frame = &audio[start..start + FRAME_LEN];

            // r[0] = sum of x[n]^2 (zero-lag autocorrelation)
            let r0: f64 = frame.iter().map(|&x| (x as f64) * (x as f64)).sum();
            if r0 < 1e-10 {
                continue; // Silent frame — skip
            }

            // Find lag with maximum normalized autocorrelation in [LAG_MIN, LAG_MAX]
            let lag_upper = LAG_MAX.min(FRAME_LEN - 1);
            let mut best_lag = LAG_MIN;
            let mut best_corr = f64::NEG_INFINITY;

            for lag in LAG_MIN..=lag_upper {
                let mut r_lag: f64 = 0.0;
                for n in 0..(FRAME_LEN - lag) {
                    r_lag += (frame[n] as f64) * (frame[n + lag] as f64);
                }
                let norm_corr = r_lag / r0;
                if norm_corr > best_corr {
                    best_corr = norm_corr;
                    best_lag = lag;
                }
            }

            if best_corr > VOICED_THRESHOLD {
                let f0 = SAMPLE_RATE / best_lag as f64;
                // Store log-F0 in semitones: 12 * log2(f0)
                if f0 > 0.0 {
                    voiced_log_f0.push(12.0 * f0.log2());
                }
            }
        }

        if voiced_log_f0.is_empty() {
            return Ok(0.0);
        }

        // Compute standard deviation of log-F0 (in semitones)
        let n = voiced_log_f0.len() as f64;
        let mean = voiced_log_f0.iter().sum::<f64>() / n;
        let variance = voiced_log_f0
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        // Normalize: divide by 12 to map semitone std_dev → [0, 1] range
        Ok((std_dev / 12.0).clamp(0.0, 1.0) as f32)
    }

    /// Build a Hann-windowed, zero-padded FFT magnitude spectrum from the
    /// centre of `audio` (up to 2048 samples, padded to 2048).
    ///
    /// Returns `(magnitudes, n_bins)` where `n_bins = N_FFT / 2 + 1`.
    fn compute_windowed_fft_magnitudes(audio: &[f32]) -> Result<(Vec<f64>, usize)> {
        const N_FFT: usize = 2048;
        let n_bins = N_FFT / 2 + 1;

        // Take up to N_FFT samples from the middle of the signal
        let (start, frame_len) = if audio.len() >= N_FFT {
            let mid = audio.len() / 2;
            let half = N_FFT / 2;
            (mid.saturating_sub(half), N_FFT)
        } else {
            (0, audio.len())
        };
        let frame = &audio[start..start + frame_len];

        // Apply Hann window and zero-pad to N_FFT
        let mut windowed = vec![0.0f64; N_FFT];
        for (i, &s) in frame.iter().enumerate() {
            let w = 0.5
                * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64
                        / (frame_len.saturating_sub(1)).max(1) as f64)
                        .cos());
            windowed[i] = s as f64 * w;
        }

        let spectrum = scirs2_fft::rfft(&windowed, Some(N_FFT))
            .map_err(|e| Error::processing(format!("FFT computation failed: {e}")))?;

        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        Ok((magnitudes, n_bins))
    }

    fn estimate_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.5);
        }

        let (magnitudes, n_bins) = Self::compute_windowed_fft_magnitudes(audio)?;

        let mut weighted_sum = 0.0f64;
        let mut magnitude_sum = 0.0f64;
        for (k, &mag) in magnitudes.iter().enumerate() {
            weighted_sum += k as f64 * mag;
            magnitude_sum += mag;
        }

        if magnitude_sum < 1e-10 {
            return Ok(0.5); // Default to mid-frequency
        }

        // Normalise bin index centroid to [0, 1]
        let centroid = (weighted_sum / magnitude_sum) / (n_bins as f64 - 1.0).max(1.0);
        Ok(centroid.clamp(0.0, 1.0) as f32)
    }

    fn estimate_spectral_rolloff(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.5);
        }

        let (magnitudes, n_bins) = Self::compute_windowed_fft_magnitudes(audio)?;

        // Power spectrum
        let power: Vec<f64> = magnitudes.iter().map(|&m| m * m).collect();
        let total_power: f64 = power.iter().sum();
        if total_power < 1e-10 {
            return Ok(0.5);
        }

        let threshold = 0.85 * total_power;
        let mut cumulative = 0.0f64;
        for (k, &p) in power.iter().enumerate() {
            cumulative += p;
            if cumulative >= threshold {
                return Ok((k as f64 / (n_bins as f64 - 1.0).max(1.0)).clamp(0.0, 1.0) as f32);
            }
        }

        Ok(1.0)
    }

    fn features_to_emotion_state(
        &self,
        features: EmotionFeatures,
    ) -> Result<voirs_emotion::types::EmotionState> {
        // Convert audio features to emotion dimensions

        // Arousal correlates with energy and pitch variation
        let arousal =
            (features.energy_level * 0.6 + features.pitch_variation * 0.4).clamp(0.0, 1.0);

        // Valence correlates with spectral characteristics and energy stability
        let energy_stability = 1.0 - features.energy_variance.min(1.0);
        let valence = (features.spectral_centroid * 0.5 + energy_stability * 0.5).clamp(0.0, 1.0);

        // Dominance correlates with overall energy and spectral rolloff
        let dominance =
            (features.peak_energy * 0.7 + features.spectral_rolloff * 0.3).clamp(0.0, 1.0);

        // Create emotion state (simplified)
        let mut emotion_state = voirs_emotion::types::EmotionState::default();
        emotion_state.current.emotion_vector.dimensions.arousal = arousal;
        emotion_state.current.emotion_vector.dimensions.valence = valence;
        emotion_state.current.emotion_vector.dimensions.dominance = dominance;
        // Note: intensity is calculated from emotion vector magnitude when needed

        Ok(emotion_state)
    }

    /// Apply a real phase-vocoder pitch shift.
    ///
    /// `pitch_factor` arrives as an additive pitch *modulation* centred at 0
    /// (`0.0` = no change, positive = higher, negative = lower). It is mapped to
    /// the phase-vocoder frequency ratio as `ratio = 1.0 + pitch_factor`
    /// (e.g. `+0.2 → 1.2×`, `+1.0 → 2.0×` ≈ one octave up). The shift reuses the
    /// WOLA phase vocoder in [`crate::transforms::PitchTransform`]
    /// (`apply_phase_vocoder_pitch_shift`: FRAME = 1024, HOP = 256, Hann
    /// analysis/synthesis windows, instantaneous-frequency bin remapping and
    /// OLA normalisation). Pitch shifting preserves length, so the result is
    /// copied back into the in-place buffer.
    fn apply_pitch_modulation(&self, audio: &mut [f32], pitch_factor: f32) -> Result<()> {
        let modulation = pitch_factor.clamp(-1.0, 1.0);
        if modulation.abs() < 0.001 {
            return Ok(());
        }
        let ratio = (1.0 + modulation).clamp(0.5, 2.0);

        // Real WOLA phase-vocoder pitch shift (output length == input length).
        let shifted = PitchTransform::new(ratio).apply(audio)?;

        let n = audio.len().min(shifted.len());
        audio[..n].copy_from_slice(&shifted[..n]);
        if n < audio.len() {
            audio[n..].fill(0.0);
        }
        Ok(())
    }

    /// Apply a real spectral-envelope (formant) shift that preserves pitch.
    ///
    /// `formant_factor` is an additive modulation centred at 0; it maps to an
    /// envelope-warp ratio `ratio = 1.0 + formant_factor` (`> 1` raises the
    /// formants, `< 1` lowers them). The algorithm is the classic homomorphic
    /// (cepstral-liftering) formant shifter — the same spectral-envelope ratio
    /// technique used elsewhere in VoiRS (cloning `apply_formant_shifts`,
    /// singing `spectral_conversion`):
    ///
    /// 1. STFT each Hann-windowed frame (FRAME = 1024, HOP = 256).
    /// 2. Estimate the smooth spectral envelope via low-quefrency cepstral
    ///    liftering: `cepstrum = irfft(log|X|)`, keep `|q| <= LIFTER`, then
    ///    `env = exp(re(rfft(liftered_cepstrum)))`.
    /// 3. Warp the envelope frequency axis by `ratio` (`warped[k] = env[k/ratio]`).
    /// 4. Re-apply to the original excitation via the per-bin magnitude ratio
    ///    `gain = warped_env / env`, so only the envelope (formants) moves while
    ///    the harmonic fine structure — and therefore the pitch — is untouched.
    /// 5. ISTFT + Hann synthesis window + overlap-add (normalised by sum w^2).
    fn apply_formant_modification(&self, audio: &mut [f32], formant_factor: f32) -> Result<()> {
        let shift = formant_factor.clamp(-0.5, 0.5);
        if shift.abs() < 0.001 {
            return Ok(());
        }
        let formant_ratio = (1.0 + shift).clamp(0.5, 2.0) as f64;

        const FRAME: usize = 1024;
        const HOP: usize = 256;
        const LIFTER: usize = 30; // cepstral lifter order -> envelope smoothness

        if audio.len() < FRAME {
            // Too short to estimate a meaningful spectral envelope.
            return Ok(());
        }

        let n_bins = FRAME / 2 + 1;
        let window: Vec<f64> = (0..FRAME)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos())
            .collect();

        let mut output = vec![0.0f64; audio.len()];
        let mut norm = vec![0.0f64; audio.len()];

        let mut start = 0usize;
        while start + FRAME <= audio.len() {
            // ── Analysis: Hann window + RFFT ─────────────────────────────
            let frame: Vec<f64> = audio[start..start + FRAME]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s as f64 * w)
                .collect();
            let spectrum = scirs2_fft::rfft(&frame, Some(FRAME))
                .map_err(|e| Error::processing(format!("formant STFT failed: {e}")))?;

            // ── Spectral envelope via cepstral liftering ─────────────────
            let log_mag: Vec<f64> = spectrum.iter().map(|c| (c.norm() + 1e-10).ln()).collect();
            // Real cepstrum (irfft of the real log-magnitude half-spectrum).
            let mut cepstrum = scirs2_fft::irfft(log_mag.as_slice(), Some(FRAME))
                .map_err(|e| Error::processing(format!("cepstrum IFFT failed: {e}")))?;
            // Low-quefrency lifter: keep the slowly varying (formant) structure.
            for (q_idx, c) in cepstrum.iter_mut().enumerate() {
                let quefrency = q_idx.min(FRAME - q_idx);
                if quefrency > LIFTER {
                    *c = 0.0;
                }
            }
            // Smoothed log-envelope = re(rfft(liftered cepstrum)); rfft∘irfft is
            // the identity for scirs2_fft, so no rescaling is required.
            let env_spec = scirs2_fft::rfft(&cepstrum, Some(FRAME))
                .map_err(|e| Error::processing(format!("envelope FFT failed: {e}")))?;
            let envelope: Vec<f64> = env_spec.iter().map(|c| c.re.exp()).collect();

            // ── Warp the envelope frequency axis by `formant_ratio` ──────
            let warped: Vec<f64> = (0..n_bins)
                .map(|k| {
                    let src = k as f64 / formant_ratio;
                    let lo = src.floor() as usize;
                    if lo + 1 >= n_bins {
                        envelope[n_bins - 1]
                    } else {
                        let frac = src - lo as f64;
                        envelope[lo] * (1.0 - frac) + envelope[lo + 1] * frac
                    }
                })
                .collect();

            // ── Re-apply warped envelope to the original excitation ──────
            let mut new_spectrum = spectrum.clone();
            for ((c, &w), &e) in new_spectrum
                .iter_mut()
                .zip(warped.iter())
                .zip(envelope.iter())
            {
                let gain = (w / e.max(1e-10)).clamp(0.1, 10.0);
                c.re *= gain;
                c.im *= gain;
            }

            // ── Synthesis: ISTFT + Hann window + overlap-add ─────────────
            let time_frame = scirs2_fft::irfft(new_spectrum.as_slice(), Some(FRAME))
                .map_err(|e| Error::processing(format!("formant ISTFT failed: {e}")))?;
            for (j, (&w, &s)) in window.iter().zip(time_frame.iter()).enumerate() {
                let idx = start + j;
                if idx < output.len() {
                    output[idx] += s * w;
                    norm[idx] += w * w;
                }
            }

            start += HOP;
        }

        // OLA normalisation (leave uncovered edge samples untouched).
        for ((s, &o), &nrm) in audio.iter_mut().zip(output.iter()).zip(norm.iter()) {
            if nrm > 1e-8 {
                *s = (o / nrm) as f32;
            }
        }

        Ok(())
    }

    /// Apply a real, pitch-preserving time-scale modification (TSM).
    ///
    /// The previous implementation merely scaled amplitude (loudness), which is
    /// not a tempo change. This reuses the phase-vocoder TSM in
    /// [`crate::transforms::SpeedTransform`]
    /// (`apply_psola_time_stretch`: analysis hop != synthesis hop, so duration
    /// changes while pitch is preserved).
    ///
    /// `rhythm_factor` is an additive modulation centred at 0; it maps to the
    /// time-scale (speed) ratio `speed = 1.0 + rhythm_factor` (`> 1` = faster /
    /// shorter, `< 1` = slower / longer).
    ///
    /// Buffer-length contract: TSM changes the sample count, but this helper
    /// operates on a fixed-length `&mut [f32]`. We therefore render the
    /// pitch-preserved, time-scaled signal and map it into the buffer by copying
    /// the leading samples and zero-padding (faster) or truncating (slower). We
    /// deliberately do *not* resample back to the original length, because that
    /// would re-introduce a pitch shift and defeat the pitch-preserving property
    /// of TSM.
    fn apply_rhythm_modification(&self, audio: &mut [f32], rhythm_factor: f32) -> Result<()> {
        let rhythm = rhythm_factor.clamp(-0.5, 0.5);
        if rhythm.abs() < 0.001 {
            return Ok(());
        }
        let speed = (1.0 + rhythm).clamp(0.25, 4.0);

        // Real pitch-preserving phase-vocoder time-scale modification.
        let stretched = SpeedTransform::new(speed).apply(audio)?;

        let n = audio.len().min(stretched.len());
        audio[..n].copy_from_slice(&stretched[..n]);
        if n < audio.len() {
            audio[n..].fill(0.0);
        }
        Ok(())
    }

    fn apply_spectral_tilt(&self, audio: &mut [f32], tilt_factor: f32) -> Result<()> {
        // Apply spectral tilt for emotional coloring
        let tilt = (tilt_factor * 0.25).clamp(-0.5, 0.5);

        if tilt.abs() < 0.001 {
            return Ok(());
        }

        // Apply high-frequency emphasis or de-emphasis
        let audio_len = audio.len();
        for (i, sample) in audio.iter_mut().enumerate() {
            let freq_weight = (i as f32 / audio_len as f32) * tilt;
            *sample *= 1.0 + freq_weight;
        }

        Ok(())
    }

    fn apply_intensity_scaling(&self, audio: &mut [f32], intensity: f32) -> Result<()> {
        // Apply overall intensity scaling based on emotional strength
        let scale_factor = 1.0 + (intensity - 0.5) * 0.3; // Scale around neutral

        for sample in audio.iter_mut() {
            *sample *= scale_factor;
            // Prevent clipping
            *sample = sample.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    fn apply_speaker_conversion_with_emotion_preservation(
        &self,
        input_audio: &[f32],
        target_characteristics: &crate::types::VoiceCharacteristics,
        source_emotion: &voirs_emotion::types::EmotionState,
    ) -> Result<Vec<f32>> {
        if input_audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        let mut converted_audio = input_audio.to_vec();

        // Apply speaker characteristics transformation while preserving emotion
        self.apply_speaker_characteristics(&mut converted_audio, target_characteristics)?;

        // Re-apply the source emotion to the converted voice
        let emotion_params = self.extract_emotion_parameters(source_emotion, 0.8);

        // Restore emotional characteristics
        self.apply_pitch_modulation(&mut converted_audio, emotion_params.pitch_modification)?;
        self.apply_formant_modification(&mut converted_audio, emotion_params.formant_shift)?;
        self.apply_rhythm_modification(&mut converted_audio, emotion_params.rhythm_adjustment)?;
        self.apply_spectral_tilt(&mut converted_audio, emotion_params.spectral_tilt)?;

        Ok(converted_audio)
    }

    fn apply_speaker_characteristics(
        &self,
        audio: &mut [f32],
        characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<()> {
        // Apply speaker-specific transformations
        // This would typically involve more sophisticated voice conversion algorithms

        // Apply gender-based modifications
        if let Some(gender) = &characteristics.gender {
            match gender {
                crate::types::Gender::Male => {
                    // Lower pitch and formants for male voice
                    self.apply_pitch_modulation(audio, -0.2)?;
                    self.apply_formant_modification(audio, -0.15)?;
                }
                crate::types::Gender::Female => {
                    // Higher pitch and formants for female voice
                    self.apply_pitch_modulation(audio, 0.2)?;
                    self.apply_formant_modification(audio, 0.15)?;
                }
                crate::types::Gender::NonBinary => {
                    // Neutral characteristics
                    self.apply_pitch_modulation(audio, 0.0)?;
                }
                crate::types::Gender::Other | crate::types::Gender::Unknown => {
                    // Neutral characteristics for unknown/other genders
                    self.apply_pitch_modulation(audio, 0.0)?;
                }
            }
        }

        // Apply age-based modifications
        if let Some(age_group) = &characteristics.age_group {
            match age_group {
                crate::types::AgeGroup::Child => {
                    // Higher pitch, more energy in high frequencies
                    self.apply_pitch_modulation(audio, 0.4)?;
                    self.apply_spectral_tilt(audio, 0.3)?;
                }
                crate::types::AgeGroup::Teen => {
                    // Slightly higher pitch than young adult
                    self.apply_pitch_modulation(audio, 0.2)?;
                    self.apply_spectral_tilt(audio, 0.1)?;
                }
                crate::types::AgeGroup::YoungAdult => {
                    // Moderate characteristics
                    self.apply_pitch_modulation(audio, 0.1)?;
                }
                crate::types::AgeGroup::Adult => {
                    // Baseline characteristics (no modification)
                }
                crate::types::AgeGroup::MiddleAged => {
                    // Slightly lower pitch than adult
                    self.apply_pitch_modulation(audio, -0.05)?;
                }
                crate::types::AgeGroup::Senior => {
                    // Lower pitch, more breathiness
                    self.apply_pitch_modulation(audio, -0.15)?;
                    self.apply_spectral_tilt(audio, -0.2)?;
                }
                crate::types::AgeGroup::Unknown => {
                    // No modification for unknown age
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "emotion-integration")]
impl Default for EmotionConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters for emotional voice modification
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionParameters {
    /// Pitch modification factor (-1.0 to 1.0)
    pub pitch_modification: f32,
    /// Formant shift factor (-1.0 to 1.0)
    pub formant_shift: f32,
    /// Rhythm adjustment factor (-1.0 to 1.0)
    pub rhythm_adjustment: f32,
    /// Spectral tilt modification (-1.0 to 1.0)
    pub spectral_tilt: f32,
}

/// Audio features used for emotion detection
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionFeatures {
    /// Average energy level
    pub energy_level: f32,
    /// Energy variance (stability)
    pub energy_variance: f32,
    /// Pitch variation measure
    pub pitch_variation: f32,
    /// Spectral centroid (brightness)
    pub spectral_centroid: f32,
    /// Spectral rolloff point
    pub spectral_rolloff: f32,
    /// Peak energy level
    pub peak_energy: f32,
}

/// Emotion transfer configuration
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionTransferConfig {
    /// Enable pitch modification
    pub enable_pitch_mod: bool,
    /// Enable formant shifting
    pub enable_formant_shift: bool,
    /// Enable rhythm adjustment
    pub enable_rhythm_adj: bool,
    /// Enable spectral tilt
    pub enable_spectral_tilt: bool,
    /// Overall intensity scaling factor
    pub intensity_scale: f32,
    /// Minimum intensity threshold
    pub min_intensity: f32,
    /// Maximum intensity threshold
    pub max_intensity: f32,
}

#[cfg(feature = "emotion-integration")]
impl Default for EmotionTransferConfig {
    fn default() -> Self {
        Self {
            enable_pitch_mod: true,
            enable_formant_shift: true,
            enable_rhythm_adj: true,
            enable_spectral_tilt: true,
            intensity_scale: 1.0,
            min_intensity: 0.1,
            max_intensity: 1.0,
        }
    }
}

// Stub implementation when emotion integration is disabled
#[cfg(not(feature = "emotion-integration"))]
#[derive(Debug, Clone)]
pub struct EmotionConversionAdapter;

#[cfg(not(feature = "emotion-integration"))]
impl EmotionConversionAdapter {
    pub fn new() -> Self {
        Self
    }

    pub async fn convert_with_emotion(
        &self,
        _input_audio: &[f32],
        _target_emotion: &EmotionState,
        _intensity: f32,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Emotion integration not enabled. Enable with 'emotion-integration' feature."
                .to_string(),
        ))
    }

    pub async fn convert_preserving_emotion(
        &self,
        _input_audio: &[f32],
        _target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Emotion integration not enabled. Enable with 'emotion-integration' feature."
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "emotion-integration"))]
impl Default for EmotionConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Stub emotion state when emotion integration is disabled
#[cfg(not(feature = "emotion-integration"))]
#[derive(Debug, Clone)]
pub struct EmotionState {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub emotion_label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_adapter_creation() {
        let adapter = EmotionConversionAdapter::new();
        assert!(matches!(adapter, EmotionConversionAdapter { .. }));
    }

    #[cfg(feature = "emotion-integration")]
    #[tokio::test]
    async fn test_emotion_conversion_validation() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let emotion = voirs_emotion::types::EmotionState::default();

        // Test invalid intensity
        let result = adapter.convert_with_emotion(&audio, &emotion, 1.5).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("intensity must be between"));

        // Test empty audio
        let result = adapter.convert_with_emotion(&[], &emotion, 0.5).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        // Test valid conversion
        let result = adapter.convert_with_emotion(&audio, &emotion, 0.5).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), audio.len());
    }

    #[cfg(feature = "emotion-integration")]
    #[tokio::test]
    async fn test_emotion_transfer_between_speakers() {
        let adapter = EmotionConversionAdapter::new();
        let source_audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let target_audio = vec![0.2, 0.3, 0.4, 0.5, 0.6];

        let result = adapter
            .transfer_emotion_between_speakers(&source_audio, &target_audio, 0.7)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), target_audio.len());

        // Test validation errors
        let result = adapter
            .transfer_emotion_between_speakers(&[], &target_audio, 0.7)
            .await;
        assert!(result.is_err());

        let result = adapter
            .transfer_emotion_between_speakers(&source_audio, &target_audio, 1.5)
            .await;
        assert!(result.is_err());
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_blending() {
        let adapter = EmotionConversionAdapter::new();
        let emotion_a = voirs_emotion::types::EmotionState::default();
        let emotion_b = voirs_emotion::types::EmotionState::default();

        let result = adapter.blend_emotions(&emotion_a, &emotion_b, 0.5);
        assert!(result.is_ok());

        // Test invalid blend ratio
        let result = adapter.blend_emotions(&emotion_a, &emotion_b, 1.5);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Blend ratio must be between"));
    }

    #[cfg(feature = "emotion-integration")]
    #[tokio::test]
    async fn test_emotion_transition() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1; 1000]; // Create longer audio for transition
        let start_emotion = voirs_emotion::types::EmotionState::default();
        let end_emotion = voirs_emotion::types::EmotionState::default();

        let result = adapter
            .apply_emotion_transition(&audio, &start_emotion, &end_emotion, 4)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), audio.len());

        // Test validation errors
        let result = adapter
            .apply_emotion_transition(&[], &start_emotion, &end_emotion, 4)
            .await;
        assert!(result.is_err());

        let result = adapter
            .apply_emotion_transition(&audio, &start_emotion, &end_emotion, 0)
            .await;
        assert!(result.is_err());
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_features_extraction() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6];

        let features = adapter.extract_emotion_features(&audio);
        assert!(features.is_ok());

        let features = features.unwrap();
        assert!(features.energy_level >= 0.0);
        assert!(features.pitch_variation >= 0.0 && features.pitch_variation <= 1.0);
        assert!(features.spectral_centroid >= 0.0 && features.spectral_centroid <= 1.0);
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_transfer_config() {
        let config = EmotionTransferConfig::default();
        assert!(config.enable_pitch_mod);
        assert!(config.enable_formant_shift);
        assert!(config.enable_rhythm_adj);
        assert!(config.enable_spectral_tilt);
        assert_eq!(config.intensity_scale, 1.0);
        assert_eq!(config.min_intensity, 0.1);
        assert_eq!(config.max_intensity, 1.0);
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_detection_from_audio() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15];

        let result = adapter.detect_source_emotion(&audio);
        assert!(result.is_ok());

        let emotion = result.unwrap();
        assert!(emotion.current.emotion_vector.dimensions.arousal >= 0.0);
        assert!(emotion.current.emotion_vector.dimensions.arousal <= 1.0);
        assert!(emotion.current.emotion_vector.dimensions.valence >= 0.0);
        assert!(emotion.current.emotion_vector.dimensions.valence <= 1.0);
        assert!(emotion.current.emotion_vector.dimensions.dominance >= 0.0);
        assert!(emotion.current.emotion_vector.dimensions.dominance <= 1.0);

        // Test empty audio error
        let result = adapter.detect_source_emotion(&[]);
        assert!(result.is_err());
    }

    // ── Real-DSP tests for pitch / formant / rhythm modification ──────────

    /// Frequency (Hz) of the largest non-DC bin of a Hann-windowed spectrum.
    #[cfg(feature = "emotion-integration")]
    fn detect_peak_frequency(signal: &[f32], sample_rate: f32) -> f32 {
        let n = signal.len();
        let windowed: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
                s as f64 * w
            })
            .collect();
        let spectrum = scirs2_fft::rfft(&windowed, Some(n)).expect("rfft");
        let mut best_bin = 1usize;
        let mut best_mag = 0.0f64;
        for (k, c) in spectrum.iter().enumerate().skip(1) {
            let m = c.norm();
            if m > best_mag {
                best_mag = m;
                best_bin = k;
            }
        }
        best_bin as f32 * sample_rate / n as f32
    }

    /// Autocorrelation pitch-period (in samples) within `[min_lag, max_lag]`.
    #[cfg(feature = "emotion-integration")]
    fn detect_fundamental_period(signal: &[f32], min_lag: usize, max_lag: usize) -> usize {
        let r0: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let r0 = r0.max(1e-12);
        let upper = max_lag.min(signal.len().saturating_sub(1));
        let mut best_lag = min_lag;
        let mut best = f64::NEG_INFINITY;
        for lag in min_lag..=upper {
            let mut r = 0.0f64;
            for n in 0..(signal.len() - lag) {
                r += signal[n] as f64 * signal[n + lag] as f64;
            }
            let norm = r / r0;
            if norm > best {
                best = norm;
                best_lag = lag;
            }
        }
        best_lag
    }

    /// Magnitude-weighted spectral centroid (Hz) of a Hann-windowed spectrum.
    #[cfg(feature = "emotion-integration")]
    fn spectral_centroid_hz(signal: &[f32], sample_rate: f32) -> f32 {
        let n = signal.len();
        let windowed: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
                s as f64 * w
            })
            .collect();
        let spectrum = scirs2_fft::rfft(&windowed, Some(n)).expect("rfft");
        let mut wsum = 0.0f64;
        let mut msum = 0.0f64;
        for (k, c) in spectrum.iter().enumerate() {
            let m = c.norm();
            wsum += k as f64 * m;
            msum += m;
        }
        if msum < 1e-12 {
            return 0.0;
        }
        (wsum / msum) as f32 * sample_rate / n as f32
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_pitch_modulation_shifts_octave_up() {
        let adapter = EmotionConversionAdapter::new();
        let sr = 22050.0f32;
        let f0 = 220.0f32;
        let n = 8192;
        let mut audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / sr).sin())
            .collect();

        let before = detect_peak_frequency(&audio, sr);
        // modulation 1.0 -> frequency ratio 2.0 (one octave up)
        adapter.apply_pitch_modulation(&mut audio, 1.0).unwrap();
        let after = detect_peak_frequency(&audio, sr);

        assert!(audio.iter().all(|x| x.is_finite()));
        assert_eq!(audio.len(), n); // pitch shift preserves length
        let ratio = after / before;
        assert!(
            (1.7..=2.3).contains(&ratio),
            "expected ~2x pitch (before={before} after={after} ratio={ratio})"
        );
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_formant_modification_preserves_pitch() {
        let adapter = EmotionConversionAdapter::new();
        let sr = 22050.0f32;
        let f0 = 150.0f32;
        let n = 8192;
        let formant_hz = 1500.0f32;
        let sigma = 700.0f32;

        // Harmonic complex with a Gaussian spectral (formant) envelope.
        let mut audio = vec![0.0f32; n];
        for k in 1..=40 {
            let fh = k as f32 * f0;
            if fh >= sr / 2.0 {
                break;
            }
            let d = fh - formant_hz;
            let amp = (-(d * d) / (2.0 * sigma * sigma)).exp();
            for (i, s) in audio.iter_mut().enumerate() {
                *s += amp * (2.0 * std::f32::consts::PI * fh * i as f32 / sr).sin();
            }
        }
        let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
        for s in audio.iter_mut() {
            *s /= peak;
        }

        let min_lag = (sr / 400.0) as usize;
        let max_lag = (sr / 80.0) as usize;
        let period_before = detect_fundamental_period(&audio, min_lag, max_lag);
        let centroid_before = spectral_centroid_hz(&audio, sr);

        // formant_factor 0.5 -> envelope warp ratio 1.5 (formants up, pitch fixed)
        adapter.apply_formant_modification(&mut audio, 0.5).unwrap();

        let period_after = detect_fundamental_period(&audio, min_lag, max_lag);
        let centroid_after = spectral_centroid_hz(&audio, sr);

        assert!(audio.iter().all(|x| x.is_finite()));
        assert_eq!(audio.len(), n);

        // F0 (autocorrelation period) preserved within ~10 %.
        let f0_before = sr / period_before as f32;
        let f0_after = sr / period_after as f32;
        assert!(
            (f0_after - f0_before).abs() / f0_before < 0.1,
            "F0 should be preserved (before={f0_before} after={f0_after})"
        );
        // Spectral envelope (formants) shifted upward.
        assert!(
            centroid_after > centroid_before * 1.02,
            "centroid should rise (before={centroid_before} after={centroid_after})"
        );
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_rhythm_modification_preserves_frequency() {
        let adapter = EmotionConversionAdapter::new();
        let sr = 22050.0f32;
        let f0 = 440.0f32;
        let n = 8192;
        let mut audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / sr).sin())
            .collect();

        let before = detect_peak_frequency(&audio, sr);
        // rhythm_factor 0.3 -> time-scale (speed) 1.3; pitch must be preserved
        adapter.apply_rhythm_modification(&mut audio, 0.3).unwrap();
        let after = detect_peak_frequency(&audio, sr);

        assert!(audio.iter().all(|x| x.is_finite()));
        assert_eq!(audio.len(), n); // in-place length contract honored
        assert!(
            (after - before).abs() < 30.0,
            "tempo change must preserve pitch (before={before} after={after})"
        );
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_dsp_outputs_finite_and_length_preserved() {
        let adapter = EmotionConversionAdapter::new();
        let sr = 22050.0f32;
        let n = 4096;
        let base: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr).sin())
            .collect();

        for &factor in &[-0.4f32, -0.1, 0.1, 0.4] {
            let mut a = base.clone();
            adapter.apply_pitch_modulation(&mut a, factor).unwrap();
            assert_eq!(a.len(), n);
            assert!(a.iter().all(|x| x.is_finite()));

            let mut b = base.clone();
            adapter.apply_formant_modification(&mut b, factor).unwrap();
            assert_eq!(b.len(), n);
            assert!(b.iter().all(|x| x.is_finite()));

            let mut c = base.clone();
            adapter.apply_rhythm_modification(&mut c, factor).unwrap();
            assert_eq!(c.len(), n);
            assert!(c.iter().all(|x| x.is_finite()));
        }
    }

    #[cfg(not(feature = "emotion-integration"))]
    #[tokio::test]
    async fn test_emotion_integration_disabled() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let emotion = EmotionState {
            valence: 0.8,
            arousal: 0.6,
            dominance: 0.7,
            emotion_label: "happy".to_string(),
        };

        let result = adapter.convert_with_emotion(&audio, &emotion, 0.5).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }
}
