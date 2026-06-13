//! Real-Time Emotion Transfer System
//!
//! This module provides advanced emotion transfer capabilities for singing synthesis,
//! enabling dynamic emotion manipulation during real-time performance.

use crate::precision_quality::functions::detect_f0_autocorr_frame;
use crate::types::{Articulation, Dynamics, Expression};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Real-time emotion transfer engine
///
/// Enables dynamic emotion transfer from reference audio to synthesized singing
#[derive(Debug)]
pub struct EmotionTransferEngine {
    /// Configuration
    config: EmotionTransferConfig,
    /// Emotion detector
    detector: EmotionDetector,
    /// Emotion interpolator
    interpolator: EmotionInterpolator,
    /// Emotion cache for performance
    emotion_cache: HashMap<String, EmotionVector>,
}

/// Emotion transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransferConfig {
    /// Enable real-time processing
    pub realtime: bool,
    /// Interpolation smoothness (0.0-1.0)
    pub smoothness: f32,
    /// Emotion intensity scaling (0.0-2.0)
    pub intensity_scale: f32,
    /// Transition duration in seconds
    pub transition_duration: f32,
    /// Enable emotion caching
    pub enable_caching: bool,
}

impl Default for EmotionTransferConfig {
    fn default() -> Self {
        Self {
            realtime: true,
            smoothness: 0.7,
            intensity_scale: 1.0,
            transition_duration: 0.5,
            enable_caching: true,
        }
    }
}

/// Multi-dimensional emotion vector
///
/// Represents emotion in a 3D space: valence (positive/negative),
/// arousal (calm/excited), and dominance (submissive/dominant)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmotionVector {
    /// Valence: -1.0 (negative) to 1.0 (positive)
    pub valence: f32,
    /// Arousal: -1.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominance: -1.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
    /// Intensity: 0.0 (none) to 1.0 (maximum)
    pub intensity: f32,
}

impl EmotionVector {
    /// Create a new emotion vector
    pub fn new(valence: f32, arousal: f32, dominance: f32, intensity: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            intensity: intensity.clamp(0.0, 1.0),
        }
    }

    /// Create a neutral emotion
    pub fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            intensity: 0.0,
        }
    }

    /// Create happy emotion
    pub fn happy() -> Self {
        Self::new(0.8, 0.6, 0.5, 0.8)
    }

    /// Create sad emotion
    pub fn sad() -> Self {
        Self::new(-0.7, -0.5, -0.3, 0.7)
    }

    /// Create angry emotion
    pub fn angry() -> Self {
        Self::new(-0.6, 0.8, 0.7, 0.9)
    }

    /// Create fearful emotion
    pub fn fearful() -> Self {
        Self::new(-0.5, 0.7, -0.6, 0.8)
    }

    /// Interpolate between two emotions
    pub fn interpolate(&self, other: &EmotionVector, alpha: f32) -> EmotionVector {
        let alpha = alpha.clamp(0.0, 1.0);
        EmotionVector {
            valence: self.valence + (other.valence - self.valence) * alpha,
            arousal: self.arousal + (other.arousal - self.arousal) * alpha,
            dominance: self.dominance + (other.dominance - self.dominance) * alpha,
            intensity: self.intensity + (other.intensity - self.intensity) * alpha,
        }
    }

    /// Calculate Euclidean distance to another emotion
    pub fn distance(&self, other: &EmotionVector) -> f32 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dv * dv + da * da + dd * dd).sqrt()
    }
}

impl Default for EmotionVector {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Emotion detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionDetectionResult {
    /// Detected emotion vector
    pub emotion: EmotionVector,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Emotion label (e.g., "happy", "sad")
    pub label: String,
    /// Per-frame emotions for temporal analysis
    pub frame_emotions: Vec<EmotionVector>,
}

/// Emotion detector for audio analysis
#[derive(Debug)]
pub struct EmotionDetector {
    /// Feature extractor configuration
    feature_config: FeatureConfig,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Frame size in samples
    pub frame_size: usize,
    /// Hop size in samples
    pub hop_size: usize,
    /// Enable spectral features
    pub enable_spectral: bool,
    /// Enable prosodic features
    pub enable_prosodic: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 512,
            enable_spectral: true,
            enable_prosodic: true,
        }
    }
}

impl EmotionDetector {
    /// Create new emotion detector
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            feature_config: config,
        }
    }

    /// Detect emotion from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, float)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Emotion detection result with confidence
    pub fn detect(&self, audio: &[f32], sample_rate: u32) -> EmotionDetectionResult {
        // Extract spectral features
        let spectral_features = self.extract_spectral_features(audio);

        // Extract prosodic features
        let prosodic_features = self.extract_prosodic_features(audio, sample_rate);

        // Combine features and classify emotion
        let emotion = self.classify_emotion(&spectral_features, &prosodic_features);

        // Analyze per-frame emotions
        let frame_emotions = self.analyze_temporal_emotions(audio);

        EmotionDetectionResult {
            emotion,
            confidence: 0.85, // Simplified confidence
            label: self.emotion_to_label(&emotion),
            frame_emotions,
        }
    }

    /// Extract spectral features from an audio frame.
    ///
    /// Computes a Hann-windowed real FFT ([`scirs2_fft::rfft`]) over the leading
    /// `frame_size` samples and derives a fixed-layout descriptor (length 5):
    ///
    /// | index | feature                              | units / range                 |
    /// |-------|--------------------------------------|-------------------------------|
    /// | 0     | mean-square energy                   | linear, `>= 0`                |
    /// | 1     | spectral centroid (normalized)       | fraction of Nyquist, `[0, 1]` |
    /// | 2     | spectral rolloff at 85 % energy      | fraction of Nyquist, `[0, 1]` |
    /// | 3     | spectral bandwidth (centroid spread) | fraction of Nyquist, `[0, 1]` |
    /// | 4     | spectral flatness (Wiener entropy)   | `[0, 1]` (1 = noise-like)     |
    ///
    /// Frequencies are normalized to the Nyquist limit so the descriptor does not
    /// depend on the sample rate. Index 0 (energy) is retained for the downstream
    /// heuristic emotion classifier.
    fn extract_spectral_features(&self, audio: &[f32]) -> Vec<f32> {
        let energy = if audio.is_empty() {
            0.0
        } else {
            audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32
        };

        let frame_len = self.feature_config.frame_size.min(audio.len());
        if frame_len < 4 {
            return vec![energy, 0.0, 0.0, 0.0, 0.0];
        }

        // Hann window to suppress spectral leakage.
        let windowed: Vec<f32> = audio[..frame_len]
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (frame_len - 1) as f32).cos());
                s * w
            })
            .collect();

        let spectrum = match scirs2_fft::rfft(&windowed, None) {
            Ok(s) => s,
            Err(_) => return vec![energy, 0.0, 0.0, 0.0, 0.0],
        };
        let n_bins = spectrum.len();
        if n_bins < 2 {
            return vec![energy, 0.0, 0.0, 0.0, 0.0];
        }

        let mags: Vec<f64> = spectrum
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();
        let total_mag: f64 = mags.iter().sum();
        if total_mag <= 1e-12 {
            return vec![energy, 0.0, 0.0, 0.0, 0.0];
        }

        // Bin `k` maps to normalized frequency `k / (n_bins - 1)` (1.0 == Nyquist).
        let max_bin = (n_bins - 1) as f64;
        let centroid: f64 = mags
            .iter()
            .enumerate()
            .map(|(k, &m)| (k as f64 / max_bin) * m)
            .sum::<f64>()
            / total_mag;

        // Magnitude-weighted spread around the centroid.
        let variance: f64 = mags
            .iter()
            .enumerate()
            .map(|(k, &m)| {
                let d = k as f64 / max_bin - centroid;
                d * d * m
            })
            .sum::<f64>()
            / total_mag;
        let bandwidth = variance.sqrt();

        // Normalized frequency below which 85 % of the magnitude lies.
        let rolloff_threshold = 0.85 * total_mag;
        let mut cumulative = 0.0f64;
        let mut rolloff_bin = max_bin;
        for (k, &m) in mags.iter().enumerate() {
            cumulative += m;
            if cumulative >= rolloff_threshold {
                rolloff_bin = k as f64;
                break;
            }
        }
        let rolloff = rolloff_bin / max_bin;

        // Spectral flatness: geometric mean / arithmetic mean of the power spectrum.
        let power: Vec<f64> = mags.iter().map(|&m| (m * m).max(1e-20)).collect();
        let log_mean = power.iter().map(|p| p.ln()).sum::<f64>() / power.len() as f64;
        let geo_mean = log_mean.exp();
        let arith_mean = power.iter().sum::<f64>() / power.len() as f64;
        let flatness = if arith_mean > 1e-20 {
            (geo_mean / arith_mean).clamp(0.0, 1.0)
        } else {
            0.0
        };

        vec![
            energy,
            centroid.clamp(0.0, 1.0) as f32,
            rolloff.clamp(0.0, 1.0) as f32,
            bandwidth.clamp(0.0, 1.0) as f32,
            flatness as f32,
        ]
    }

    /// Extract prosodic features from an audio frame.
    ///
    /// Frames the signal and runs the crate autocorrelation F0 detector
    /// ([`detect_f0_autocorr_frame`]) over each frame to form an F0 contour, then
    /// derives pitch, intensity and rhythm descriptors (length 7):
    ///
    /// | index | feature                          | units                  |
    /// |-------|----------------------------------|------------------------|
    /// | 0     | peak amplitude                   | linear, `[0, 1]`       |
    /// | 1     | mean F0 over voiced frames       | Hz (`0` if unvoiced)   |
    /// | 2     | F0 standard deviation            | Hz                     |
    /// | 3     | F0 range (max - min, voiced)     | Hz                     |
    /// | 4     | mean short-time intensity (RMS)  | linear                 |
    /// | 5     | intensity standard deviation     | linear                 |
    /// | 6     | onset rate                       | onsets per second      |
    ///
    /// Index 0 (peak amplitude) is retained for the downstream heuristic classifier.
    fn extract_prosodic_features(&self, audio: &[f32], sample_rate: u32) -> Vec<f32> {
        let peak_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if audio.is_empty() || sample_rate == 0 {
            return vec![peak_amplitude, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        }

        let frame_size = self.feature_config.frame_size.max(256);
        let hop_size = self.feature_config.hop_size.max(1);

        let mut f0_values: Vec<f32> = Vec::new();
        let mut intensities: Vec<f32> = Vec::new();

        let mut pos = 0;
        while pos < audio.len() {
            let end = (pos + frame_size).min(audio.len());
            let frame = &audio[pos..end];
            if frame.len() >= 64 {
                let f0 = robust_frame_f0(frame, sample_rate as f32);
                if f0 > 0.0 {
                    f0_values.push(f0);
                }
            }
            let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            intensities.push(rms);
            if end == audio.len() {
                break;
            }
            pos += hop_size;
        }

        // F0 statistics over voiced frames.
        let (f0_mean, f0_std, f0_range) = if f0_values.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mean = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
            let var =
                f0_values.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / f0_values.len() as f32;
            let min = f0_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = f0_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (mean, var.sqrt(), max - min)
        };

        // Intensity-envelope statistics.
        let (intensity_mean, intensity_std) = if intensities.is_empty() {
            (0.0, 0.0)
        } else {
            let mean = intensities.iter().sum::<f32>() / intensities.len() as f32;
            let var = intensities.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
                / intensities.len() as f32;
            (mean, var.sqrt())
        };

        // Onset rate: count rising intensity transitions crossing an adaptive
        // threshold, normalized by duration -> a speaking/syllable-rate proxy.
        let onset_rate = if intensities.len() >= 2 {
            let threshold = intensity_mean.max(1e-4) * 0.5;
            let mut onsets = 0usize;
            for w in intensities.windows(2) {
                if w[0] <= threshold && w[1] > threshold {
                    onsets += 1;
                }
            }
            let duration = audio.len() as f32 / sample_rate as f32;
            if duration > 0.0 {
                onsets as f32 / duration
            } else {
                0.0
            }
        } else {
            0.0
        };

        vec![
            peak_amplitude,
            f0_mean,
            f0_std,
            f0_range,
            intensity_mean,
            intensity_std,
            onset_rate,
        ]
    }

    /// Classify emotion from features
    fn classify_emotion(&self, spectral: &[f32], prosodic: &[f32]) -> EmotionVector {
        // Simplified emotion classification
        // In production, this would use a trained neural network
        let energy = spectral[0];
        let amplitude = prosodic[0];

        let arousal = (energy * 2.0 - 1.0).clamp(-1.0, 1.0);
        let valence = (amplitude - 0.5).clamp(-1.0, 1.0);
        let dominance = 0.0;
        let intensity = energy.min(1.0);

        EmotionVector::new(valence, arousal, dominance, intensity)
    }

    /// Analyze temporal emotion patterns
    fn analyze_temporal_emotions(&self, audio: &[f32]) -> Vec<EmotionVector> {
        let num_frames = audio.len() / self.feature_config.hop_size;
        let mut frame_emotions = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.feature_config.hop_size;
            let end = (start + self.feature_config.frame_size).min(audio.len());
            let frame = &audio[start..end];

            let spectral = self.extract_spectral_features(frame);
            let prosodic = self.extract_prosodic_features(frame, 44100);
            let emotion = self.classify_emotion(&spectral, &prosodic);

            frame_emotions.push(emotion);
        }

        frame_emotions
    }

    /// Convert emotion vector to label
    fn emotion_to_label(&self, emotion: &EmotionVector) -> String {
        if emotion.valence > 0.5 && emotion.arousal > 0.3 {
            "happy".to_string()
        } else if emotion.valence < -0.5 && emotion.arousal < 0.0 {
            "sad".to_string()
        } else if emotion.valence < -0.3 && emotion.arousal > 0.5 {
            "angry".to_string()
        } else if emotion.arousal > 0.5 && emotion.dominance < -0.3 {
            "fearful".to_string()
        } else {
            "neutral".to_string()
        }
    }
}

impl Default for EmotionDetector {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// Emotion interpolator for smooth transitions
#[derive(Debug)]
pub struct EmotionInterpolator {
    /// Smoothing window size
    window_size: usize,
    /// Previous emotions for smoothing
    emotion_history: Vec<EmotionVector>,
}

impl EmotionInterpolator {
    /// Create new emotion interpolator
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            emotion_history: Vec::new(),
        }
    }

    /// Smooth emotion transition
    ///
    /// # Arguments
    /// * `current` - Current emotion
    /// * `target` - Target emotion
    /// * `transition_progress` - Progress (0.0-1.0)
    ///
    /// # Returns
    /// Smoothed emotion vector
    pub fn smooth_transition(
        &mut self,
        current: &EmotionVector,
        target: &EmotionVector,
        transition_progress: f32,
    ) -> EmotionVector {
        // Apply smoothing curve (ease-in-out)
        let smoothed_progress = self.ease_in_out(transition_progress);

        // Interpolate
        let interpolated = current.interpolate(target, smoothed_progress);

        // Add to history
        self.emotion_history.push(interpolated);
        if self.emotion_history.len() > self.window_size {
            self.emotion_history.remove(0);
        }

        // Return moving average
        self.moving_average()
    }

    /// Apply ease-in-out curve
    fn ease_in_out(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
        }
    }

    /// Calculate moving average of emotion history
    fn moving_average(&self) -> EmotionVector {
        if self.emotion_history.is_empty() {
            return EmotionVector::neutral();
        }

        let len = self.emotion_history.len() as f32;
        let avg_valence = self.emotion_history.iter().map(|e| e.valence).sum::<f32>() / len;
        let avg_arousal = self.emotion_history.iter().map(|e| e.arousal).sum::<f32>() / len;
        let avg_dominance = self
            .emotion_history
            .iter()
            .map(|e| e.dominance)
            .sum::<f32>()
            / len;
        let avg_intensity = self
            .emotion_history
            .iter()
            .map(|e| e.intensity)
            .sum::<f32>()
            / len;

        EmotionVector::new(avg_valence, avg_arousal, avg_dominance, avg_intensity)
    }

    /// Clear emotion history
    pub fn reset(&mut self) {
        self.emotion_history.clear();
    }
}

impl Default for EmotionInterpolator {
    fn default() -> Self {
        Self::new(5)
    }
}

impl EmotionTransferEngine {
    /// Create new emotion transfer engine
    pub fn new(config: EmotionTransferConfig) -> Self {
        Self {
            config,
            detector: EmotionDetector::default(),
            interpolator: EmotionInterpolator::default(),
            emotion_cache: HashMap::new(),
        }
    }

    /// Transfer emotion from reference audio to synthesis parameters
    ///
    /// # Arguments
    /// * `reference_audio` - Reference audio samples
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Detected emotion vector
    pub fn detect_reference_emotion(
        &mut self,
        reference_audio: &[f32],
        sample_rate: u32,
    ) -> crate::Result<EmotionVector> {
        // Check cache if enabled
        let cache_key = format!("{:?}", reference_audio.len());
        if self.config.enable_caching {
            if let Some(cached_emotion) = self.emotion_cache.get(&cache_key) {
                return Ok(*cached_emotion);
            }
        }

        // Detect emotion
        let detection = self.detector.detect(reference_audio, sample_rate);
        let mut emotion = detection.emotion;

        // Apply intensity scaling
        emotion.intensity *= self.config.intensity_scale;
        emotion.intensity = emotion.intensity.clamp(0.0, 1.0);

        // Cache result
        if self.config.enable_caching {
            self.emotion_cache.insert(cache_key, emotion);
        }

        Ok(emotion)
    }

    /// Apply emotion to synthesis parameters
    ///
    /// # Arguments
    /// * `emotion` - Emotion vector to apply
    ///
    /// # Returns
    /// Modified synthesis parameters
    pub fn apply_emotion(&self, emotion: &EmotionVector) -> EmotionSynthesisParams {
        // Convert emotion to synthesis parameters
        EmotionSynthesisParams {
            tempo_scale: 1.0 + emotion.arousal * 0.2,
            pitch_shift: emotion.valence * 2.0,
            dynamics: self.emotion_to_dynamics(emotion),
            articulation: self.emotion_to_articulation(emotion),
            vibrato_rate: (1.0 + emotion.arousal * 0.3).max(0.5),
            vibrato_depth: emotion.intensity * 0.05,
        }
    }

    /// Convert emotion to dynamics
    fn emotion_to_dynamics(&self, emotion: &EmotionVector) -> Dynamics {
        let intensity = emotion.intensity;

        if intensity > 0.8 {
            Dynamics::Forte
        } else if intensity > 0.6 {
            Dynamics::MezzoForte
        } else if intensity > 0.4 {
            Dynamics::MezzoPiano
        } else {
            Dynamics::Piano
        }
    }

    /// Convert emotion to articulation
    fn emotion_to_articulation(&self, emotion: &EmotionVector) -> Articulation {
        if emotion.arousal > 0.5 {
            Articulation::Staccato
        } else if emotion.valence < -0.5 {
            Articulation::Tenuto
        } else {
            Articulation::Normal
        }
    }

    /// Smooth emotion transition
    pub fn smooth_transition(
        &mut self,
        current: &EmotionVector,
        target: &EmotionVector,
        progress: f32,
    ) -> EmotionVector {
        self.interpolator
            .smooth_transition(current, target, progress)
    }

    /// Reset the emotion transfer engine
    pub fn reset(&mut self) {
        self.interpolator.reset();
        self.emotion_cache.clear();
    }
}

impl Default for EmotionTransferEngine {
    fn default() -> Self {
        Self::new(EmotionTransferConfig::default())
    }
}

/// Synthesis parameters derived from emotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionSynthesisParams {
    /// Tempo scaling factor
    pub tempo_scale: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Dynamics level
    pub dynamics: Dynamics,
    /// Articulation style
    pub articulation: Articulation,
    /// Vibrato rate in Hz
    pub vibrato_rate: f32,
    /// Vibrato depth (0.0-1.0)
    pub vibrato_depth: f32,
}

/// Normalized autocorrelation coefficient of a frame at `lag`, clamped to `[0, 1]`.
fn normalized_autocorr(frame: &[f32], lag: usize) -> f32 {
    if lag == 0 || lag >= frame.len() {
        return 0.0;
    }
    let mean = frame.iter().sum::<f32>() / frame.len() as f32;
    let n = frame.len() - lag;
    let mut cross = 0.0f32;
    let mut left = 0.0f32;
    let mut right = 0.0f32;
    for i in 0..n {
        let a = frame[i] - mean;
        let b = frame[i + lag] - mean;
        cross += a * b;
        left += a * a;
        right += b * b;
    }
    let denom = (left * right).sqrt();
    if denom > 1e-12 {
        (cross / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Robust per-frame F0 (Hz) via autocorrelation with octave-down correction.
///
/// Wraps the crate detector [`detect_f0_autocorr_frame`], which maximizes a *raw*
/// autocorrelation and can therefore lock onto an integer multiple of the true
/// period (an octave-down error). The frame is peak-normalized so the detector's
/// raw threshold is gain-robust, and when a shorter sub-period correlates at
/// least as well, the higher (fundamental) frequency is preferred. This mirrors
/// the proven approach used by the zero-shot voice analyzer.
fn robust_frame_f0(frame: &[f32], sample_rate: f32) -> f32 {
    let peak = frame.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak <= 1e-9 {
        return 0.0;
    }
    let normalized: Vec<f32> = frame.iter().map(|&x| x / peak).collect();

    let coarse = detect_f0_autocorr_frame(&normalized, sample_rate);
    if coarse <= 0.0 {
        return 0.0;
    }
    let period = (sample_rate / coarse).round() as usize;
    if period < 2 {
        return coarse;
    }
    let min_lag = ((sample_rate / 800.0) as usize).max(2);
    let base = normalized_autocorr(&normalized, period);
    let mut best_period = period;
    for div in [4usize, 3, 2] {
        let candidate = period / div;
        if candidate >= min_lag
            && normalized_autocorr(&normalized, candidate) >= 0.85 * base.max(1e-6)
        {
            best_period = candidate;
            break;
        }
    }
    sample_rate / best_period as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_vector_creation() {
        let emotion = EmotionVector::new(0.8, 0.6, 0.5, 0.9);
        assert_eq!(emotion.valence, 0.8);
        assert_eq!(emotion.arousal, 0.6);
        assert_eq!(emotion.dominance, 0.5);
        assert_eq!(emotion.intensity, 0.9);
    }

    #[test]
    fn test_emotion_vector_clamping() {
        let emotion = EmotionVector::new(2.0, -2.0, 1.5, -0.5);
        assert_eq!(emotion.valence, 1.0);
        assert_eq!(emotion.arousal, -1.0);
        assert_eq!(emotion.dominance, 1.0);
        assert_eq!(emotion.intensity, 0.0);
    }

    #[test]
    fn test_predefined_emotions() {
        let happy = EmotionVector::happy();
        assert!(happy.valence > 0.0);
        assert!(happy.intensity > 0.0);

        let sad = EmotionVector::sad();
        assert!(sad.valence < 0.0);

        let angry = EmotionVector::angry();
        assert!(angry.arousal > 0.0);
    }

    #[test]
    fn test_emotion_interpolation() {
        let happy = EmotionVector::happy();
        let sad = EmotionVector::sad();

        let mid = happy.interpolate(&sad, 0.5);
        assert!(mid.valence < happy.valence);
        assert!(mid.valence > sad.valence);
    }

    #[test]
    fn test_emotion_distance() {
        let happy = EmotionVector::happy();
        let sad = EmotionVector::sad();

        let distance = happy.distance(&sad);
        assert!(distance > 1.0);

        let same_distance = happy.distance(&happy);
        assert_eq!(same_distance, 0.0);
    }

    #[test]
    fn test_emotion_detector() {
        let detector = EmotionDetector::default();
        let audio = vec![0.5; 44100]; // 1 second of audio

        let result = detector.detect(&audio, 44100);
        assert!(result.confidence > 0.0);
        assert!(!result.label.is_empty());
        assert!(!result.frame_emotions.is_empty());
    }

    #[test]
    fn test_emotion_interpolator() {
        let mut interpolator = EmotionInterpolator::default();
        let current = EmotionVector::neutral();
        let target = EmotionVector::happy();

        let smoothed = interpolator.smooth_transition(&current, &target, 0.5);
        assert!(smoothed.valence >= 0.0);
    }

    #[test]
    fn test_emotion_transfer_engine() {
        let mut engine = EmotionTransferEngine::default();
        let audio = vec![0.5; 44100];

        let emotion = engine.detect_reference_emotion(&audio, 44100).unwrap();
        assert!(emotion.intensity >= 0.0);

        let params = engine.apply_emotion(&emotion);
        assert!(params.tempo_scale > 0.0);
        assert!(params.vibrato_rate > 0.0);
    }

    #[test]
    fn test_emotion_caching() {
        let config = EmotionTransferConfig {
            enable_caching: true,
            ..Default::default()
        };
        let mut engine = EmotionTransferEngine::new(config);

        let audio = vec![0.5; 44100];
        let emotion1 = engine.detect_reference_emotion(&audio, 44100).unwrap();
        let emotion2 = engine.detect_reference_emotion(&audio, 44100).unwrap();

        // Should return same result from cache
        assert_eq!(emotion1.valence, emotion2.valence);
    }

    #[test]
    fn test_smooth_transition() {
        let mut engine = EmotionTransferEngine::default();
        let current = EmotionVector::neutral();
        let target = EmotionVector::happy();

        let smoothed = engine.smooth_transition(&current, &target, 0.5);
        assert!(smoothed.valence > current.valence);
        assert!(smoothed.valence < target.valence);
    }

    #[test]
    fn test_emotion_to_label() {
        let detector = EmotionDetector::default();

        let happy = EmotionVector::happy();
        assert_eq!(detector.emotion_to_label(&happy), "happy");

        let sad = EmotionVector::sad();
        assert_eq!(detector.emotion_to_label(&sad), "sad");

        let angry = EmotionVector::angry();
        assert_eq!(detector.emotion_to_label(&angry), "angry");
    }

    #[test]
    fn test_config_defaults() {
        let config = EmotionTransferConfig::default();
        assert!(config.realtime);
        assert_eq!(config.smoothness, 0.7);
        assert_eq!(config.intensity_scale, 1.0);
    }

    /// Generate a steady sinusoid of the given frequency.
    fn make_tone(freq: f32, sample_rate: u32, secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.8 * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Deterministic, dependency-free white noise via a PCG-style LCG.
    fn make_white_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Top 31 bits -> [0, 1), mapped to the symmetric range [-1, 1).
                let u = (state >> 33) as f32 / (1u64 << 31) as f32;
                u * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_spectral_features_tone_vs_noise() {
        let sr = 44100;
        let detector = EmotionDetector::default();

        // Low pure tone: peaked spectrum -> low centroid, low flatness.
        let tone = make_tone(300.0, sr, 0.1);
        // Broadband noise: flat spectrum -> mid centroid, high flatness.
        let noise = make_white_noise(4096, 0x1234_5678);

        let sf_tone = detector.extract_spectral_features(&tone);
        let sf_noise = detector.extract_spectral_features(&noise);

        assert_eq!(sf_tone.len(), 5);
        assert_eq!(sf_noise.len(), 5);

        // Flatness: noise is far closer to 1.0 than a tone.
        assert!(
            sf_noise[4] > sf_tone[4] + 0.2,
            "flatness tone {} vs noise {}",
            sf_tone[4],
            sf_noise[4]
        );
        // Centroid: broadband noise sits well above a low 300 Hz tone.
        assert!(
            sf_noise[1] > sf_tone[1],
            "centroid tone {} vs noise {}",
            sf_tone[1],
            sf_noise[1]
        );
        // A low tone's normalized centroid stays near the bottom of the band.
        assert!(sf_tone[1] < 0.2, "low-tone centroid {}", sf_tone[1]);
    }

    #[test]
    fn test_prosodic_f0_synthetic_tone() {
        let sr = 44100;
        let detector = EmotionDetector::default();
        let tone = make_tone(220.0, sr, 0.5);

        let pf = detector.extract_prosodic_features(&tone, sr);
        assert_eq!(pf.len(), 7);

        // Mean F0 should match the synthesized 220 Hz tone.
        assert!((pf[1] - 220.0).abs() < 15.0, "mean F0 {}", pf[1]);
        // A steady tone has near-zero F0 spread.
        assert!(pf[2] < 20.0, "F0 std {}", pf[2]);
        // A continuous tone produces essentially no syllable onsets.
        assert!(pf[6] < 5.0, "onset rate {}", pf[6]);
    }
}
