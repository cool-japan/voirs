//! F0 (Fundamental Frequency) tracking implementations
//!
//! This module provides multiple algorithms for fundamental frequency extraction:
//! - RAPT (Robust Algorithm for Pitch Tracking)
//! - YIN (Autocorrelation-based algorithm)
//! - SWIPE (Sawtooth Waveform Inspired Pitch Estimator)
//!
//! These algorithms are essential for prosody analysis, voice quality assessment,
//! and speech naturalness evaluation.

use crate::EvaluationResult;
use voirs_sdk::AudioBuffer;

/// F0 tracking configuration
#[derive(Debug, Clone)]
pub struct F0TrackingConfig {
    /// Minimum F0 frequency in Hz
    pub f0_min: f32,
    /// Maximum F0 frequency in Hz
    pub f0_max: f32,
    /// Frame length in seconds
    pub frame_length: f32,
    /// Frame hop in seconds
    pub frame_hop: f32,
    /// Voicing threshold
    pub voicing_threshold: f32,
    /// Octave cost for SWIPE
    pub octave_cost: f32,
    /// Voicing transition cost
    pub voicing_transition_cost: f32,
}

impl Default for F0TrackingConfig {
    fn default() -> Self {
        Self {
            f0_min: 75.0,       // Typical male voice minimum
            f0_max: 600.0,      // Typical female voice maximum
            frame_length: 0.04, // 40ms frames
            frame_hop: 0.01,    // 10ms hop
            voicing_threshold: 0.45,
            octave_cost: 0.01,
            voicing_transition_cost: 0.005,
        }
    }
}

/// F0 tracking result for a single frame
#[derive(Debug, Clone, PartialEq)]
pub struct F0Frame {
    /// Time in seconds
    pub time: f32,
    /// F0 frequency in Hz (None if unvoiced)
    pub f0: Option<f32>,
    /// Voicing probability [0.0, 1.0]
    pub voicing_probability: f32,
    /// Confidence in the F0 estimate [0.0, 1.0]
    pub confidence: f32,
}

/// Complete F0 contour
#[derive(Debug, Clone)]
pub struct F0Contour {
    /// F0 frames
    pub frames: Vec<F0Frame>,
    /// Sample rate of original audio
    pub sample_rate: u32,
    /// Configuration used for extraction
    pub config: F0TrackingConfig,
    /// Algorithm used
    pub algorithm: F0Algorithm,
}

/// Available F0 tracking algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum F0Algorithm {
    /// RAPT (Robust Algorithm for Pitch Tracking)
    RAPT,
    /// YIN algorithm
    YIN,
    /// SWIPE algorithm
    SWIPE,
    /// Autocorrelation-based simple algorithm
    Autocorrelation,
}

/// F0 tracker implementation
pub struct F0Tracker {
    config: F0TrackingConfig,
    algorithm: F0Algorithm,
}

impl F0Tracker {
    /// Create new F0 tracker with specified algorithm
    #[must_use]
    pub fn new(algorithm: F0Algorithm, config: F0TrackingConfig) -> Self {
        Self { config, algorithm }
    }

    /// Create RAPT tracker with default config
    #[must_use]
    pub fn rapt() -> Self {
        Self::new(F0Algorithm::RAPT, F0TrackingConfig::default())
    }

    /// Create YIN tracker with default config
    #[must_use]
    pub fn yin() -> Self {
        Self::new(F0Algorithm::YIN, F0TrackingConfig::default())
    }

    /// Create SWIPE tracker with default config
    #[must_use]
    pub fn swipe() -> Self {
        Self::new(F0Algorithm::SWIPE, F0TrackingConfig::default())
    }

    /// Create autocorrelation tracker with default config
    #[must_use]
    pub fn autocorr() -> Self {
        Self::new(F0Algorithm::Autocorrelation, F0TrackingConfig::default())
    }

    /// Extract F0 contour from audio
    pub async fn extract_f0(&self, audio: &AudioBuffer) -> EvaluationResult<F0Contour> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        let frame_length_samples = (self.config.frame_length * sample_rate) as usize;
        let frame_hop_samples = (self.config.frame_hop * sample_rate) as usize;

        let mut frames = Vec::new();
        let mut pos = 0;

        while pos + frame_length_samples <= samples.len() {
            let frame_samples = &samples[pos..pos + frame_length_samples];
            let time = pos as f32 / sample_rate;

            let f0_frame = match self.algorithm {
                F0Algorithm::RAPT => self.extract_f0_rapt(frame_samples, sample_rate).await?,
                F0Algorithm::YIN => self.extract_f0_yin(frame_samples, sample_rate).await?,
                F0Algorithm::SWIPE => self.extract_f0_swipe(frame_samples, sample_rate).await?,
                F0Algorithm::Autocorrelation => {
                    self.extract_f0_autocorr(frame_samples, sample_rate).await?
                }
            };

            frames.push(F0Frame {
                time,
                f0: f0_frame.f0,
                voicing_probability: f0_frame.voicing_probability,
                confidence: f0_frame.confidence,
            });

            pos += frame_hop_samples;
        }

        Ok(F0Contour {
            frames,
            sample_rate: audio.sample_rate(),
            config: self.config.clone(),
            algorithm: self.algorithm,
        })
    }

    /// RAPT algorithm implementation
    async fn extract_f0_rapt(&self, frame: &[f32], sample_rate: f32) -> EvaluationResult<F0Frame> {
        // Simplified RAPT implementation
        // Real RAPT involves complex correlation analysis and dynamic programming

        // First pass: rough F0 estimation using NCCF
        let nccf = self.compute_normalized_cross_correlation(frame, sample_rate)?;
        let rough_f0 = self.find_f0_from_nccf(&nccf, sample_rate)?;

        // Second pass: refine F0 using local search
        let refined_f0 = if let Some(f0) = rough_f0 {
            self.refine_f0_rapt(frame, f0, sample_rate)?
        } else {
            None
        };

        // Calculate voicing probability
        let voicing_prob = self.calculate_voicing_probability_rapt(frame, refined_f0)?;

        Ok(F0Frame {
            time: 0.0, // Will be set by caller
            f0: refined_f0,
            voicing_probability: voicing_prob,
            confidence: if refined_f0.is_some() { 0.8 } else { 0.2 },
        })
    }

    /// YIN algorithm implementation
    async fn extract_f0_yin(&self, frame: &[f32], sample_rate: f32) -> EvaluationResult<F0Frame> {
        // YIN difference function
        let yin_buffer = self.compute_yin_difference_function(frame)?;

        // Cumulative mean normalized difference function
        let cmdf = self.compute_cmdf(&yin_buffer)?;

        // Find absolute threshold crossing
        let period = self.find_yin_period(&cmdf)?;

        let f0 = if let Some(p) = period {
            if p > 0.0 && p < frame.len() as f32 {
                Some(sample_rate / p)
            } else {
                None
            }
        } else {
            None
        };

        // Voicing based on YIN threshold
        let voicing_prob = if let Some(p) = period {
            let index = p.round() as usize;
            if index < cmdf.len() {
                1.0 - cmdf[index]
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(F0Frame {
            time: 0.0,
            f0: f0.filter(|&f| f >= self.config.f0_min && f <= self.config.f0_max),
            voicing_probability: voicing_prob,
            confidence: if f0.is_some() { 0.85 } else { 0.15 },
        })
    }

    /// SWIPE algorithm implementation
    async fn extract_f0_swipe(&self, frame: &[f32], sample_rate: f32) -> EvaluationResult<F0Frame> {
        // SWIPE uses sawtooth waves as basis functions
        let candidates = self.generate_f0_candidates()?;
        let mut best_strength = 0.0;
        let mut best_f0 = None;

        for &candidate_f0 in &candidates {
            let strength = self.compute_swipe_strength(frame, candidate_f0, sample_rate)?;

            if strength > best_strength {
                best_strength = strength;
                best_f0 = Some(candidate_f0);
            }
        }

        // Apply voicing threshold
        let is_voiced = best_strength > self.config.voicing_threshold;
        let f0 = if is_voiced { best_f0 } else { None };

        Ok(F0Frame {
            time: 0.0,
            f0,
            voicing_probability: best_strength,
            confidence: if f0.is_some() { 0.75 } else { 0.25 },
        })
    }

    /// Simple autocorrelation-based F0 extraction
    async fn extract_f0_autocorr(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> EvaluationResult<F0Frame> {
        let autocorr = self.compute_autocorrelation(frame)?;

        // Find peaks in autocorrelation
        let min_period = (sample_rate / self.config.f0_max) as usize;
        let max_period = (sample_rate / self.config.f0_min) as usize;

        let mut best_period = None;
        let mut max_value = 0.0;

        for period in min_period..max_period.min(autocorr.len() - 1) {
            if autocorr[period] > max_value {
                max_value = autocorr[period];
                best_period = Some(period);
            }
        }

        let f0 = best_period.map(|p| sample_rate / p as f32);
        let voicing_prob = max_value.max(0.0).min(1.0);

        Ok(F0Frame {
            time: 0.0,
            f0,
            voicing_probability: voicing_prob,
            confidence: if f0.is_some() { 0.7 } else { 0.3 },
        })
    }

    // Helper methods for F0 algorithms

    fn compute_normalized_cross_correlation(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> EvaluationResult<Vec<f32>> {
        let min_period = (sample_rate / self.config.f0_max) as usize;
        let max_period = (sample_rate / self.config.f0_min) as usize;

        let mut nccf = vec![0.0; max_period - min_period + 1];

        for (i, lag) in (min_period..=max_period).enumerate() {
            if lag >= frame.len() {
                break;
            }

            let mut correlation = 0.0;
            let mut energy1 = 0.0;
            let mut energy2 = 0.0;

            for j in 0..(frame.len() - lag) {
                correlation += frame[j] * frame[j + lag];
                energy1 += frame[j] * frame[j];
                energy2 += frame[j + lag] * frame[j + lag];
            }

            if energy1 > 0.0 && energy2 > 0.0 {
                nccf[i] = correlation / (energy1 * energy2).sqrt();
            }
        }

        Ok(nccf)
    }

    fn find_f0_from_nccf(&self, nccf: &[f32], sample_rate: f32) -> EvaluationResult<Option<f32>> {
        let mut max_idx = 0;
        let mut max_val = 0.0;

        for (i, &val) in nccf.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val > self.config.voicing_threshold {
            let min_period = (sample_rate / self.config.f0_max) as usize;
            let period = min_period + max_idx;
            Ok(Some(sample_rate / period as f32))
        } else {
            Ok(None)
        }
    }

    fn refine_f0_rapt(
        &self,
        _frame: &[f32],
        f0: f32,
        _sample_rate: f32,
    ) -> EvaluationResult<Option<f32>> {
        // Simplified refinement - in real RAPT this involves interpolation
        Ok(Some(f0))
    }

    fn calculate_voicing_probability_rapt(
        &self,
        frame: &[f32],
        f0: Option<f32>,
    ) -> EvaluationResult<f32> {
        if f0.is_none() {
            return Ok(0.0);
        }

        // Simple voicing probability based on signal characteristics
        let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        let zero_crossings = frame.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let zcr = zero_crossings as f32 / frame.len() as f32;

        // High energy and low zero-crossing rate suggest voicing
        let energy_score = (energy * 10.0).min(1.0);
        let zcr_score = (1.0 - zcr * 10.0).max(0.0);

        Ok((energy_score + zcr_score) / 2.0)
    }

    fn compute_yin_difference_function(&self, frame: &[f32]) -> EvaluationResult<Vec<f32>> {
        let n = frame.len();
        let max_lag = n / 2;
        let mut diff = vec![0.0; max_lag];

        for lag in 1..max_lag {
            let mut sum = 0.0;
            for j in 0..(n - lag) {
                let d = frame[j] - frame[j + lag];
                sum += d * d;
            }
            diff[lag] = sum;
        }

        Ok(diff)
    }

    fn compute_cmdf(&self, diff: &[f32]) -> EvaluationResult<Vec<f32>> {
        let mut cmdf = vec![1.0; diff.len()];
        let mut running_sum = 0.0;

        for i in 1..diff.len() {
            running_sum += diff[i];
            if running_sum > 0.0 {
                cmdf[i] = diff[i] / (running_sum / i as f32);
            }
        }

        Ok(cmdf)
    }

    fn find_yin_period(&self, cmdf: &[f32]) -> EvaluationResult<Option<f32>> {
        // Find first minimum below threshold
        for (i, &val) in cmdf.iter().enumerate().skip(1) {
            if val < self.config.voicing_threshold {
                // Parabolic interpolation for sub-sample accuracy
                if i > 0 && i < cmdf.len() - 1 {
                    let x0 = cmdf[i - 1];
                    let x1 = cmdf[i];
                    let x2 = cmdf[i + 1];

                    let a = (x0 - 2.0 * x1 + x2) / 2.0;
                    if a.abs() > 1e-10 {
                        let correction = (x2 - x0) / (4.0 * a);
                        return Ok(Some(i as f32 + correction));
                    }
                }
                return Ok(Some(i as f32));
            }
        }
        Ok(None)
    }

    fn generate_f0_candidates(&self) -> EvaluationResult<Vec<f32>> {
        let mut candidates = Vec::new();
        let mut f0 = self.config.f0_min;

        while f0 <= self.config.f0_max {
            candidates.push(f0);
            f0 *= 1.01; // Small logarithmic steps
        }

        Ok(candidates)
    }

    fn compute_swipe_strength(
        &self,
        frame: &[f32],
        f0: f32,
        sample_rate: f32,
    ) -> EvaluationResult<f32> {
        let period = sample_rate / f0;
        let period_samples = period as usize;

        if period_samples == 0 || period_samples >= frame.len() {
            return Ok(0.0);
        }

        // Simplified SWIPE strength calculation
        let mut strength = 0.0;
        let mut count = 0;

        for h in 1..=5 {
            // Check first 5 harmonics
            let harmonic_period = period_samples / h;
            if harmonic_period < frame.len() {
                let mut correlation = 0.0;
                for i in 0..(frame.len() - harmonic_period) {
                    correlation += frame[i] * frame[i + harmonic_period];
                }
                strength += correlation.abs();
                count += 1;
            }
        }

        if count > 0 {
            Ok(strength / count as f32)
        } else {
            Ok(0.0)
        }
    }

    fn compute_autocorrelation(&self, frame: &[f32]) -> EvaluationResult<Vec<f32>> {
        let n = frame.len();
        let mut autocorr = vec![0.0; n];

        for lag in 0..n {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += frame[i] * frame[i + lag];
            }
            autocorr[lag] = sum;
        }

        // Normalize by lag 0
        let norm_factor = autocorr[0];
        if norm_factor > 0.0 {
            for val in &mut autocorr {
                *val /= norm_factor;
            }
        }

        Ok(autocorr)
    }
}

impl F0Contour {
    /// Get F0 values as a vector (None values become NaN)
    #[must_use]
    pub fn f0_values(&self) -> Vec<f32> {
        self.frames
            .iter()
            .map(|frame| frame.f0.unwrap_or(f32::NAN))
            .collect()
    }

    /// Get time values
    #[must_use]
    pub fn time_values(&self) -> Vec<f32> {
        self.frames.iter().map(|frame| frame.time).collect()
    }

    /// Get voicing decisions
    #[must_use]
    pub fn voicing_decisions(&self) -> Vec<bool> {
        self.frames
            .iter()
            .map(|frame| frame.voicing_probability > 0.5)
            .collect()
    }

    /// Calculate F0 statistics
    #[must_use]
    pub fn statistics(&self) -> F0Statistics {
        let voiced_f0s: Vec<f32> = self.frames.iter().filter_map(|frame| frame.f0).collect();

        if voiced_f0s.is_empty() {
            return F0Statistics::default();
        }

        let mean = voiced_f0s.iter().sum::<f32>() / voiced_f0s.len() as f32;
        let variance = voiced_f0s
            .iter()
            .map(|&f0| (f0 - mean).powi(2))
            .sum::<f32>()
            / voiced_f0s.len() as f32;
        let std_dev = variance.sqrt();

        let min = voiced_f0s.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = voiced_f0s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let voiced_frames = self
            .frames
            .iter()
            .filter(|frame| frame.f0.is_some())
            .count();
        let voicing_rate = voiced_frames as f32 / self.frames.len() as f32;

        F0Statistics {
            mean,
            std_dev,
            min,
            max,
            voicing_rate,
            total_frames: self.frames.len(),
            voiced_frames,
        }
    }

    /// Smooth F0 contour using median filtering
    pub fn smooth(&mut self, window_size: usize) {
        if window_size < 3 || window_size % 2 == 0 {
            return; // Invalid window size
        }

        let half_window = window_size / 2;
        let mut smoothed_f0s = Vec::new();

        for i in 0..self.frames.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(self.frames.len());

            let mut window_f0s: Vec<f32> = self.frames[start..end]
                .iter()
                .filter_map(|frame| frame.f0)
                .collect();

            if window_f0s.is_empty() {
                smoothed_f0s.push(None);
            } else {
                window_f0s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = window_f0s[window_f0s.len() / 2];
                smoothed_f0s.push(Some(median));
            }
        }

        for (frame, &smoothed_f0) in self.frames.iter_mut().zip(smoothed_f0s.iter()) {
            frame.f0 = smoothed_f0;
        }
    }
}

/// F0 contour statistics
#[derive(Debug, Clone, PartialEq)]
pub struct F0Statistics {
    /// Mean F0 of voiced frames
    pub mean: f32,
    /// Standard deviation of F0
    pub std_dev: f32,
    /// Minimum F0 value
    pub min: f32,
    /// Maximum F0 value
    pub max: f32,
    /// Fraction of voiced frames
    pub voicing_rate: f32,
    /// Total number of frames
    pub total_frames: usize,
    /// Number of voiced frames
    pub voiced_frames: usize,
}

impl Default for F0Statistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            voicing_rate: 0.0,
            total_frames: 0,
            voiced_frames: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn create_test_audio(sample_rate: u32, duration: f32, f0: f32) -> AudioBuffer {
        let num_samples = (sample_rate as f32 * duration) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * PI * f0 * t).sin() * 0.5;
            samples.push(sample);
        }

        AudioBuffer::new(samples, sample_rate, 1)
    }

    #[tokio::test]
    async fn test_f0_tracker_creation() {
        let tracker = F0Tracker::rapt();
        assert_eq!(tracker.algorithm, F0Algorithm::RAPT);

        let tracker = F0Tracker::yin();
        assert_eq!(tracker.algorithm, F0Algorithm::YIN);

        let tracker = F0Tracker::swipe();
        assert_eq!(tracker.algorithm, F0Algorithm::SWIPE);
    }

    #[tokio::test]
    async fn test_f0_extraction_rapt() {
        let tracker = F0Tracker::rapt();
        let audio = create_test_audio(16000, 1.0, 200.0);

        let contour = tracker.extract_f0(&audio).await.unwrap();

        assert!(!contour.frames.is_empty());
        assert_eq!(contour.algorithm, F0Algorithm::RAPT);
        assert_eq!(contour.sample_rate, 16000);

        // Check that we detected some F0 values
        let voiced_frames = contour.frames.iter().filter(|f| f.f0.is_some()).count();
        assert!(voiced_frames > 0);
    }

    #[tokio::test]
    async fn test_f0_extraction_yin() {
        let tracker = F0Tracker::yin();
        let audio = create_test_audio(16000, 1.0, 150.0);

        let contour = tracker.extract_f0(&audio).await.unwrap();

        assert!(!contour.frames.is_empty());
        assert_eq!(contour.algorithm, F0Algorithm::YIN);

        // YIN should detect F0 for synthetic sine wave
        let voiced_frames = contour.frames.iter().filter(|f| f.f0.is_some()).count();
        assert!(voiced_frames > 0);
    }

    #[tokio::test]
    async fn test_f0_extraction_swipe() {
        let tracker = F0Tracker::swipe();
        let audio = create_test_audio(16000, 1.0, 300.0);

        let contour = tracker.extract_f0(&audio).await.unwrap();

        assert!(!contour.frames.is_empty());
        assert_eq!(contour.algorithm, F0Algorithm::SWIPE);
    }

    #[tokio::test]
    async fn test_f0_contour_statistics() {
        let tracker = F0Tracker::autocorr();
        let audio = create_test_audio(16000, 0.5, 220.0);

        let contour = tracker.extract_f0(&audio).await.unwrap();
        let stats = contour.statistics();

        assert!(stats.total_frames > 0);
        if stats.voiced_frames > 0 {
            assert!(stats.mean > 0.0);
            assert!(stats.voicing_rate >= 0.0 && stats.voicing_rate <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_f0_contour_smoothing() {
        let tracker = F0Tracker::yin();
        let audio = create_test_audio(16000, 0.5, 180.0);

        let mut contour = tracker.extract_f0(&audio).await.unwrap();
        let original_frames = contour.frames.len();

        contour.smooth(5);

        // Should have same number of frames after smoothing
        assert_eq!(contour.frames.len(), original_frames);
    }

    #[tokio::test]
    async fn test_silent_audio() {
        let tracker = F0Tracker::rapt();
        let audio = AudioBuffer::new(vec![0.0; 16000], 16000, 1);

        let contour = tracker.extract_f0(&audio).await.unwrap();

        // Silent audio should have low voicing probabilities
        let high_voicing_frames = contour
            .frames
            .iter()
            .filter(|f| f.voicing_probability > 0.7)
            .count();

        assert!(high_voicing_frames < contour.frames.len() / 2);
    }

    #[tokio::test]
    async fn test_f0_tracking_config() {
        let mut config = F0TrackingConfig::default();
        config.f0_min = 100.0;
        config.f0_max = 400.0;
        config.voicing_threshold = 0.3;

        let tracker = F0Tracker::new(F0Algorithm::YIN, config.clone());
        let audio = create_test_audio(16000, 0.5, 250.0);

        let contour = tracker.extract_f0(&audio).await.unwrap();

        assert_eq!(contour.config.f0_min, 100.0);
        assert_eq!(contour.config.f0_max, 400.0);
        assert_eq!(contour.config.voicing_threshold, 0.3);
    }
}
