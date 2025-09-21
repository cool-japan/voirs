//! Voiced/Unvoiced (VUV) analysis and evaluation
//!
//! This module provides comprehensive VUV decision analysis, including:
//! - Multiple VUV detection algorithms
//! - VUV decision accuracy evaluation
//! - Temporal VUV pattern analysis
//! - Cross-algorithm VUV comparison

use crate::EvaluationResult;
use voirs_sdk::AudioBuffer;

/// VUV analysis configuration
#[derive(Debug, Clone)]
pub struct VuvConfig {
    /// Frame length in seconds
    pub frame_length: f32,
    /// Frame hop in seconds
    pub frame_hop: f32,
    /// Energy threshold for voicing decision
    pub energy_threshold: f32,
    /// Zero-crossing rate threshold
    pub zcr_threshold: f32,
    /// Autocorrelation threshold
    pub autocorr_threshold: f32,
    /// Spectral centroid threshold
    pub spectral_centroid_threshold: f32,
    /// Spectral rolloff threshold
    pub spectral_rolloff_threshold: f32,
    /// Voicing probability threshold
    pub voicing_prob_threshold: f32,
}

impl Default for VuvConfig {
    fn default() -> Self {
        Self {
            frame_length: 0.025,     // 25ms frames
            frame_hop: 0.01,         // 10ms hop
            energy_threshold: -30.0, // dB energy threshold (more sensitive)
            zcr_threshold: 0.3,
            autocorr_threshold: 0.3,
            spectral_centroid_threshold: 1000.0,
            spectral_rolloff_threshold: 0.85,
            voicing_prob_threshold: 0.5,
        }
    }
}

/// VUV decision for a single frame
#[derive(Debug, Clone, PartialEq)]
pub struct VuvFrame {
    /// Time in seconds
    pub time: f32,
    /// Voiced decision (true = voiced, false = unvoiced)
    pub is_voiced: bool,
    /// Voicing probability [0.0, 1.0]
    pub voicing_probability: f32,
    /// Individual feature values
    pub features: VuvFeatures,
}

/// VUV feature values for analysis
#[derive(Debug, Clone, PartialEq)]
pub struct VuvFeatures {
    /// Energy (log scale)
    pub energy: f32,
    /// Zero-crossing rate
    pub zcr: f32,
    /// Autocorrelation peak
    pub autocorr_peak: f32,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Spectral rolloff (ratio)
    pub spectral_rolloff: f32,
    /// Harmonic-to-noise ratio
    pub hnr: f32,
    /// Spectral flatness
    pub spectral_flatness: f32,
}

/// Complete VUV analysis result
#[derive(Debug, Clone)]
pub struct VuvAnalysis {
    /// VUV frames
    pub frames: Vec<VuvFrame>,
    /// Sample rate of original audio
    pub sample_rate: u32,
    /// Configuration used
    pub config: VuvConfig,
    /// Algorithm used
    pub algorithm: VuvAlgorithm,
    /// Overall statistics
    pub statistics: VuvStatistics,
}

/// Available VUV detection algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VuvAlgorithm {
    /// Energy-based decision
    Energy,
    /// Zero-crossing rate based
    ZeroCrossing,
    /// Autocorrelation based
    Autocorrelation,
    /// Spectral features based
    SpectralFeatures,
    /// Multi-feature combination
    MultiFature,
    /// Machine learning based (simplified)
    MachineLearning,
}

/// VUV analysis statistics
#[derive(Debug, Clone, PartialEq)]
pub struct VuvStatistics {
    /// Total number of frames
    pub total_frames: usize,
    /// Number of voiced frames
    pub voiced_frames: usize,
    /// Number of unvoiced frames
    pub unvoiced_frames: usize,
    /// Voicing rate (fraction of voiced frames)
    pub voicing_rate: f32,
    /// Average voicing probability for voiced frames
    pub avg_voiced_probability: f32,
    /// Average voicing probability for unvoiced frames
    pub avg_unvoiced_probability: f32,
    /// Longest voiced segment (frames)
    pub longest_voiced_segment: usize,
    /// Longest unvoiced segment (frames)
    pub longest_unvoiced_segment: usize,
    /// Number of voicing transitions
    pub voicing_transitions: usize,
}

/// VUV comparison result
#[derive(Debug, Clone)]
pub struct VuvComparison {
    /// Reference VUV analysis
    pub reference: VuvAnalysis,
    /// Test VUV analysis
    pub test: VuvAnalysis,
    /// Accuracy metrics
    pub accuracy: VuvAccuracy,
    /// Temporal alignment
    pub alignment: VuvAlignment,
}

/// VUV accuracy metrics
#[derive(Debug, Clone, PartialEq)]
pub struct VuvAccuracy {
    /// Overall accuracy (correct decisions / total decisions)
    pub overall_accuracy: f32,
    /// Voiced accuracy (correctly identified voiced frames / total voiced frames)
    pub voiced_accuracy: f32,
    /// Unvoiced accuracy (correctly identified unvoiced frames / total unvoiced frames)
    pub unvoiced_accuracy: f32,
    /// Precision for voiced detection
    pub voiced_precision: f32,
    /// Recall for voiced detection
    pub voiced_recall: f32,
    /// F1 score for voiced detection
    pub voiced_f1: f32,
    /// False positive rate (unvoiced identified as voiced)
    pub false_positive_rate: f32,
    /// False negative rate (voiced identified as unvoiced)
    pub false_negative_rate: f32,
    /// Correlation coefficient between voicing probabilities
    pub probability_correlation: f32,
}

/// VUV temporal alignment analysis
#[derive(Debug, Clone, PartialEq)]
pub struct VuvAlignment {
    /// Alignment accuracy
    pub alignment_accuracy: f32,
    /// Average time shift (seconds)
    pub avg_time_shift: f32,
    /// Segment boundary accuracy
    pub boundary_accuracy: f32,
    /// Transition timing errors
    pub transition_errors: Vec<f32>,
}

/// VUV analyzer implementation
pub struct VuvAnalyzer {
    config: VuvConfig,
}

impl VuvAnalyzer {
    /// Create a new VUV analyzer
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(VuvConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: VuvConfig) -> Self {
        Self { config }
    }

    /// Analyze VUV decisions in audio
    pub async fn analyze(
        &self,
        audio: &AudioBuffer,
        algorithm: VuvAlgorithm,
    ) -> EvaluationResult<VuvAnalysis> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        let frame_length_samples = (self.config.frame_length * sample_rate) as usize;
        let frame_hop_samples = (self.config.frame_hop * sample_rate) as usize;

        let mut frames = Vec::new();
        let mut pos = 0;

        while pos + frame_length_samples <= samples.len() {
            let frame_samples = &samples[pos..pos + frame_length_samples];
            let time = pos as f32 / sample_rate;

            let features = self.extract_features(frame_samples, sample_rate).await?;
            let (is_voiced, voicing_probability) = self.make_vuv_decision(&features, algorithm)?;

            frames.push(VuvFrame {
                time,
                is_voiced,
                voicing_probability,
                features,
            });

            pos += frame_hop_samples;
        }

        let statistics = self.calculate_statistics(&frames);

        Ok(VuvAnalysis {
            frames,
            sample_rate: audio.sample_rate(),
            config: self.config.clone(),
            algorithm,
            statistics,
        })
    }

    /// Compare VUV decisions between reference and test audio
    pub async fn compare(
        &self,
        reference: &AudioBuffer,
        test: &AudioBuffer,
        algorithm: VuvAlgorithm,
    ) -> EvaluationResult<VuvComparison> {
        let ref_analysis = self.analyze(reference, algorithm).await?;
        let test_analysis = self.analyze(test, algorithm).await?;

        let accuracy = self.calculate_accuracy(&ref_analysis, &test_analysis)?;
        let alignment = self.analyze_alignment(&ref_analysis, &test_analysis)?;

        Ok(VuvComparison {
            reference: ref_analysis,
            test: test_analysis,
            accuracy,
            alignment,
        })
    }

    /// Extract VUV features from audio frame
    async fn extract_features(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> EvaluationResult<VuvFeatures> {
        let energy = self.calculate_energy(frame);
        let zcr = self.calculate_zcr(frame);
        let autocorr_peak = self.calculate_autocorr_peak(frame)?;
        let spectral_centroid = self.calculate_spectral_centroid(frame, sample_rate).await?;
        let spectral_rolloff = self.calculate_spectral_rolloff(frame, sample_rate).await?;
        let hnr = self.calculate_hnr(frame)?;
        let spectral_flatness = self.calculate_spectral_flatness(frame).await?;

        Ok(VuvFeatures {
            energy,
            zcr,
            autocorr_peak,
            spectral_centroid,
            spectral_rolloff,
            hnr,
            spectral_flatness,
        })
    }

    /// Make VUV decision based on features and algorithm
    fn make_vuv_decision(
        &self,
        features: &VuvFeatures,
        algorithm: VuvAlgorithm,
    ) -> EvaluationResult<(bool, f32)> {
        let voicing_prob = match algorithm {
            VuvAlgorithm::Energy => {
                // Energy is dB scale, higher values are more voiced

                if features.energy > self.config.energy_threshold {
                    ((features.energy - self.config.energy_threshold) / 30.0)
                        .min(1.0)
                        .max(0.0)
                } else {
                    0.0
                }
            }
            VuvAlgorithm::ZeroCrossing => {
                let zcr_score = 1.0 - (features.zcr / self.config.zcr_threshold).min(1.0);
                zcr_score.max(0.0)
            }
            VuvAlgorithm::Autocorrelation => (features.autocorr_peak
                / self.config.autocorr_threshold)
                .min(1.0)
                .max(0.0),
            VuvAlgorithm::SpectralFeatures => {
                let centroid_score =
                    if features.spectral_centroid < self.config.spectral_centroid_threshold {
                        1.0 - features.spectral_centroid / self.config.spectral_centroid_threshold
                    } else {
                        0.0
                    };
                let rolloff_score = features.spectral_rolloff;
                let flatness_score = 1.0 - features.spectral_flatness;

                (centroid_score + rolloff_score + flatness_score) / 3.0
            }
            VuvAlgorithm::MultiFature => {
                // Combine multiple features with weights
                let energy_score = if features.energy > self.config.energy_threshold {
                    ((features.energy - self.config.energy_threshold) / 30.0)
                        .min(1.0)
                        .max(0.0)
                } else {
                    0.0
                };
                let zcr_score = 1.0 - (features.zcr / self.config.zcr_threshold).min(1.0).max(0.0);
                let autocorr_score = (features.autocorr_peak / self.config.autocorr_threshold)
                    .min(1.0)
                    .max(0.0);
                let hnr_score = (features.hnr / 10.0).min(1.0).max(0.0); // Assume 10 dB as good HNR

                // Weighted combination
                0.3 * energy_score + 0.2 * zcr_score + 0.3 * autocorr_score + 0.2 * hnr_score
            }
            VuvAlgorithm::MachineLearning => {
                // Simplified ML-like decision using feature combination
                let feature_vector = [
                    features.energy,
                    features.zcr,
                    features.autocorr_peak,
                    features.spectral_centroid / 1000.0, // Normalize
                    features.spectral_rolloff,
                    features.hnr / 20.0, // Normalize
                    features.spectral_flatness,
                ];

                // Simple linear combination (could be replaced with actual ML model)
                let weights = [0.2, -0.15, 0.25, -0.1, 0.15, 0.2, -0.15];
                let score: f32 = feature_vector
                    .iter()
                    .zip(weights.iter())
                    .map(|(f, w)| f * w)
                    .sum();

                // Apply sigmoid to get probability
                1.0 / (1.0 + (-score).exp())
            }
        };

        let is_voiced = voicing_prob > self.config.voicing_prob_threshold;
        Ok((is_voiced, voicing_prob))
    }

    /// Calculate frame energy
    fn calculate_energy(&self, frame: &[f32]) -> f32 {
        let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        if energy > 1e-10 {
            10.0 * energy.log10() // Energy in dB scale
        } else {
            -100.0 // Very low energy
        }
    }

    /// Calculate zero-crossing rate
    fn calculate_zcr(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let crossings = frame.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Calculate autocorrelation peak
    fn calculate_autocorr_peak(&self, frame: &[f32]) -> EvaluationResult<f32> {
        if frame.is_empty() {
            return Ok(0.0);
        }

        let max_lag = frame.len() / 4; // Check up to 1/4 of frame length
        let mut max_corr: f32 = 0.0;

        for lag in 1..max_lag {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(frame.len() - lag) {
                correlation += frame[i] * frame[i + lag];
                norm1 += frame[i] * frame[i];
                norm2 += frame[i + lag] * frame[i + lag];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                let normalized_corr = correlation / (norm1 * norm2).sqrt();
                max_corr = max_corr.max(normalized_corr);
            }
        }

        Ok(max_corr)
    }

    /// Calculate spectral centroid
    async fn calculate_spectral_centroid(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> EvaluationResult<f32> {
        let spectrum = self.compute_spectrum(frame).await?;

        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let frequency = i as f32 * sample_rate / (2.0 * spectrum.len() as f32);
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate spectral rolloff
    async fn calculate_spectral_rolloff(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> EvaluationResult<f32> {
        let spectrum = self.compute_spectrum(frame).await?;

        let total_energy: f32 = spectrum.iter().sum();
        let rolloff_threshold = 0.85 * total_energy;

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude;
            if cumulative_energy >= rolloff_threshold {
                let frequency = i as f32 * sample_rate / (2.0 * spectrum.len() as f32);
                return Ok(frequency / (sample_rate / 2.0)); // Normalized to Nyquist
            }
        }

        Ok(1.0) // Full spectrum
    }

    /// Calculate harmonic-to-noise ratio
    fn calculate_hnr(&self, frame: &[f32]) -> EvaluationResult<f32> {
        if frame.is_empty() {
            return Ok(0.0);
        }

        // Simplified HNR calculation
        let signal_power = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;

        // Estimate noise power using high-frequency components
        let noise_estimate = if frame.len() > 10 {
            let differences: Vec<f32> = frame.windows(2).map(|w| w[1] - w[0]).collect();
            differences.iter().map(|x| x * x).sum::<f32>() / differences.len() as f32
        } else {
            signal_power * 0.1 // Assume 10% noise
        };

        if noise_estimate > 0.0 && signal_power > noise_estimate {
            let hnr = 10.0 * (signal_power / noise_estimate).log10();
            Ok(hnr.max(0.0).min(30.0)) // Clamp to reasonable range
        } else {
            Ok(0.0)
        }
    }

    /// Calculate spectral flatness
    async fn calculate_spectral_flatness(&self, frame: &[f32]) -> EvaluationResult<f32> {
        let spectrum = self.compute_spectrum(frame).await?;

        if spectrum.iter().any(|&x| x <= 0.0) {
            return Ok(0.0);
        }

        let geometric_mean = spectrum.iter().map(|x| x.ln()).sum::<f32>() / spectrum.len() as f32;
        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            Ok(geometric_mean.exp() / arithmetic_mean)
        } else {
            Ok(0.0)
        }
    }

    /// Compute power spectrum
    async fn compute_spectrum(&self, frame: &[f32]) -> EvaluationResult<Vec<f32>> {
        // Simplified power spectrum computation
        let n = frame.len();
        let mut spectrum = vec![0.0; n / 2 + 1];

        // Apply window
        let windowed: Vec<f32> = frame
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
                x * window
            })
            .collect();

        // Simplified DFT for magnitude spectrum
        for k in 0..spectrum.len() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in windowed.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        Ok(spectrum)
    }

    /// Calculate VUV statistics
    fn calculate_statistics(&self, frames: &[VuvFrame]) -> VuvStatistics {
        let total_frames = frames.len();
        let voiced_frames = frames.iter().filter(|f| f.is_voiced).count();
        let unvoiced_frames = total_frames - voiced_frames;

        let voicing_rate = if total_frames > 0 {
            voiced_frames as f32 / total_frames as f32
        } else {
            0.0
        };

        let avg_voiced_probability = if voiced_frames > 0 {
            frames
                .iter()
                .filter(|f| f.is_voiced)
                .map(|f| f.voicing_probability)
                .sum::<f32>()
                / voiced_frames as f32
        } else {
            0.0
        };

        let avg_unvoiced_probability = if unvoiced_frames > 0 {
            frames
                .iter()
                .filter(|f| !f.is_voiced)
                .map(|f| f.voicing_probability)
                .sum::<f32>()
                / unvoiced_frames as f32
        } else {
            0.0
        };

        // Calculate segment lengths and transitions
        let (longest_voiced, longest_unvoiced, transitions) = self.analyze_segments(frames);

        VuvStatistics {
            total_frames,
            voiced_frames,
            unvoiced_frames,
            voicing_rate,
            avg_voiced_probability,
            avg_unvoiced_probability,
            longest_voiced_segment: longest_voiced,
            longest_unvoiced_segment: longest_unvoiced,
            voicing_transitions: transitions,
        }
    }

    /// Analyze VUV segments and transitions
    fn analyze_segments(&self, frames: &[VuvFrame]) -> (usize, usize, usize) {
        if frames.is_empty() {
            return (0, 0, 0);
        }

        let mut longest_voiced = 0;
        let mut longest_unvoiced = 0;
        let mut transitions = 0;

        let mut current_voiced_length = 0;
        let mut current_unvoiced_length = 0;
        let mut last_state = frames[0].is_voiced;

        for frame in frames {
            if frame.is_voiced != last_state {
                transitions += 1;
                last_state = frame.is_voiced;
            }

            if frame.is_voiced {
                current_voiced_length += 1;
                longest_unvoiced = longest_unvoiced.max(current_unvoiced_length);
                current_unvoiced_length = 0;
            } else {
                current_unvoiced_length += 1;
                longest_voiced = longest_voiced.max(current_voiced_length);
                current_voiced_length = 0;
            }
        }

        // Final segments
        longest_voiced = longest_voiced.max(current_voiced_length);
        longest_unvoiced = longest_unvoiced.max(current_unvoiced_length);

        (longest_voiced, longest_unvoiced, transitions)
    }

    /// Calculate VUV accuracy metrics
    fn calculate_accuracy(
        &self,
        reference: &VuvAnalysis,
        test: &VuvAnalysis,
    ) -> EvaluationResult<VuvAccuracy> {
        let min_frames = reference.frames.len().min(test.frames.len());

        if min_frames == 0 {
            return Ok(VuvAccuracy {
                overall_accuracy: 0.0,
                voiced_accuracy: 0.0,
                unvoiced_accuracy: 0.0,
                voiced_precision: 0.0,
                voiced_recall: 0.0,
                voiced_f1: 0.0,
                false_positive_rate: 0.0,
                false_negative_rate: 0.0,
                probability_correlation: 0.0,
            });
        }

        let mut correct_total = 0;
        let mut ref_voiced = 0;
        let mut test_voiced = 0;
        let mut correct_voiced = 0;
        let mut correct_unvoiced = 0;
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        let mut ref_probs = Vec::new();
        let mut test_probs = Vec::new();

        for i in 0..min_frames {
            let ref_frame = &reference.frames[i];
            let test_frame = &test.frames[i];

            ref_probs.push(ref_frame.voicing_probability);
            test_probs.push(test_frame.voicing_probability);

            if ref_frame.is_voiced == test_frame.is_voiced {
                correct_total += 1;
                if ref_frame.is_voiced {
                    correct_voiced += 1;
                    true_positives += 1;
                } else {
                    correct_unvoiced += 1;
                }
            }

            if ref_frame.is_voiced {
                ref_voiced += 1;
                if !test_frame.is_voiced {
                    false_negatives += 1;
                }
            }

            if test_frame.is_voiced {
                test_voiced += 1;
                if !ref_frame.is_voiced {
                    false_positives += 1;
                }
            }
        }

        let overall_accuracy = correct_total as f32 / min_frames as f32;

        let voiced_accuracy = if ref_voiced > 0 {
            correct_voiced as f32 / ref_voiced as f32
        } else {
            1.0
        };

        let unvoiced_accuracy = if (min_frames - ref_voiced) > 0 {
            correct_unvoiced as f32 / (min_frames - ref_voiced) as f32
        } else {
            1.0
        };

        let voiced_precision = if test_voiced > 0 {
            true_positives as f32 / test_voiced as f32
        } else {
            0.0
        };

        let voiced_recall = if ref_voiced > 0 {
            true_positives as f32 / ref_voiced as f32
        } else {
            0.0
        };

        let voiced_f1 = if voiced_precision + voiced_recall > 0.0 {
            2.0 * voiced_precision * voiced_recall / (voiced_precision + voiced_recall)
        } else {
            0.0
        };

        let false_positive_rate = if (min_frames - ref_voiced) > 0 {
            false_positives as f32 / (min_frames - ref_voiced) as f32
        } else {
            0.0
        };

        let false_negative_rate = if ref_voiced > 0 {
            false_negatives as f32 / ref_voiced as f32
        } else {
            0.0
        };

        let probability_correlation = crate::calculate_correlation(&ref_probs, &test_probs);

        Ok(VuvAccuracy {
            overall_accuracy,
            voiced_accuracy,
            unvoiced_accuracy,
            voiced_precision,
            voiced_recall,
            voiced_f1,
            false_positive_rate,
            false_negative_rate,
            probability_correlation,
        })
    }

    /// Analyze temporal alignment
    fn analyze_alignment(
        &self,
        reference: &VuvAnalysis,
        test: &VuvAnalysis,
    ) -> EvaluationResult<VuvAlignment> {
        // Simplified alignment analysis
        let min_frames = reference.frames.len().min(test.frames.len());

        if min_frames == 0 {
            return Ok(VuvAlignment {
                alignment_accuracy: 0.0,
                avg_time_shift: 0.0,
                boundary_accuracy: 0.0,
                transition_errors: Vec::new(),
            });
        }

        // Find voicing transitions in both sequences
        let ref_transitions = self.find_transitions(&reference.frames);
        let test_transitions = self.find_transitions(&test.frames);

        // Calculate boundary accuracy and timing errors
        let mut boundary_matches = 0;
        let mut time_errors = Vec::new();

        for ref_transition in &ref_transitions {
            if let Some(closest_test) = test_transitions
                .iter()
                .min_by(|a, b| (a.abs_diff(*ref_transition)).cmp(&(b.abs_diff(*ref_transition))))
            {
                let time_error =
                    (*closest_test as f32 - *ref_transition as f32) * self.config.frame_hop;
                time_errors.push(time_error.abs());

                // Consider a match if within 2 frames
                if closest_test.abs_diff(*ref_transition) <= 2 {
                    boundary_matches += 1;
                }
            }
        }

        let boundary_accuracy = if ref_transitions.is_empty() {
            1.0
        } else {
            boundary_matches as f32 / ref_transitions.len() as f32
        };

        let avg_time_shift = if time_errors.is_empty() {
            0.0
        } else {
            time_errors.iter().sum::<f32>() / time_errors.len() as f32
        };

        Ok(VuvAlignment {
            alignment_accuracy: boundary_accuracy,
            avg_time_shift,
            boundary_accuracy,
            transition_errors: time_errors,
        })
    }

    /// Find voicing transition frames
    fn find_transitions(&self, frames: &[VuvFrame]) -> Vec<usize> {
        if frames.len() < 2 {
            return Vec::new();
        }

        let mut transitions = Vec::new();

        for i in 1..frames.len() {
            if frames[i].is_voiced != frames[i - 1].is_voiced {
                transitions.push(i);
            }
        }

        transitions
    }
}

impl Default for VuvAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    use voirs_sdk::AudioBuffer;

    fn create_test_audio(sample_rate: u32, duration: f32, is_voiced: bool) -> AudioBuffer {
        let num_samples = (sample_rate as f32 * duration) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = if is_voiced {
                // Voiced: harmonic signal at 200 Hz
                (2.0 * PI * 200.0 * t).sin() * 0.5
            } else {
                // Unvoiced: pseudo-random noise-like signal
                let pseudo_random = ((i * 1103515245 + 12345) % 2147483648) as f32 / 2147483648.0;
                (pseudo_random - 0.5) * 0.2
            };
            samples.push(sample);
        }

        AudioBuffer::new(samples, sample_rate, 1)
    }

    #[tokio::test]
    async fn test_vuv_analyzer_creation() {
        let analyzer = VuvAnalyzer::new();
        assert_eq!(analyzer.config.frame_length, 0.025);
        assert_eq!(analyzer.config.frame_hop, 0.01);
    }

    #[tokio::test]
    async fn test_vuv_analysis_voiced() {
        let analyzer = VuvAnalyzer::new();
        let audio = create_test_audio(16000, 0.5, true); // Voiced audio

        let analysis = analyzer
            .analyze(&audio, VuvAlgorithm::Energy)
            .await
            .unwrap();

        assert!(!analysis.frames.is_empty());
        assert_eq!(analysis.algorithm, VuvAlgorithm::Energy);
        // For a strong harmonic signal, expect at least some voicing
        assert!(
            analysis.statistics.voicing_rate > 0.3,
            "Voicing rate was {}, expected > 0.3",
            analysis.statistics.voicing_rate
        );
    }

    #[tokio::test]
    async fn test_vuv_analysis_unvoiced() {
        let analyzer = VuvAnalyzer::new();
        let audio = create_test_audio(16000, 0.5, false); // Unvoiced audio

        let analysis = analyzer
            .analyze(&audio, VuvAlgorithm::ZeroCrossing)
            .await
            .unwrap();

        assert!(!analysis.frames.is_empty());
        assert_eq!(analysis.algorithm, VuvAlgorithm::ZeroCrossing);
        assert!(analysis.statistics.voicing_rate < 0.5); // Most frames should be unvoiced
    }

    #[tokio::test]
    async fn test_vuv_comparison() {
        let analyzer = VuvAnalyzer::new();
        let voiced_audio = create_test_audio(16000, 0.5, true);
        let unvoiced_audio = create_test_audio(16000, 0.5, false);

        let comparison = analyzer
            .compare(&voiced_audio, &unvoiced_audio, VuvAlgorithm::MultiFature)
            .await
            .unwrap();

        assert!(comparison.accuracy.overall_accuracy >= 0.0);
        assert!(comparison.accuracy.overall_accuracy <= 1.0);
        assert!(comparison.accuracy.voiced_precision >= 0.0);
        assert!(comparison.accuracy.voiced_recall >= 0.0);
    }

    #[tokio::test]
    async fn test_vuv_features_extraction() {
        let analyzer = VuvAnalyzer::new();
        let frame = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1]; // Small test frame

        let features = analyzer.extract_features(&frame, 16000.0).await.unwrap();

        assert!(features.energy.is_finite());
        assert!(features.zcr >= 0.0 && features.zcr <= 1.0);
        assert!(features.autocorr_peak >= 0.0 && features.autocorr_peak <= 1.0);
        assert!(features.spectral_centroid >= 0.0);
        assert!(features.spectral_rolloff >= 0.0 && features.spectral_rolloff <= 1.0);
    }

    #[tokio::test]
    async fn test_vuv_algorithms() {
        let analyzer = VuvAnalyzer::new();
        let audio = create_test_audio(16000, 0.2, true);

        // Test all algorithms
        let algorithms = [
            VuvAlgorithm::Energy,
            VuvAlgorithm::ZeroCrossing,
            VuvAlgorithm::Autocorrelation,
            VuvAlgorithm::SpectralFeatures,
            VuvAlgorithm::MultiFature,
            VuvAlgorithm::MachineLearning,
        ];

        for algorithm in algorithms {
            let analysis = analyzer.analyze(&audio, algorithm).await.unwrap();
            assert_eq!(analysis.algorithm, algorithm);
            assert!(!analysis.frames.is_empty());
        }
    }

    #[tokio::test]
    async fn test_vuv_statistics() {
        let analyzer = VuvAnalyzer::new();
        let audio = create_test_audio(16000, 0.3, true);

        let analysis = analyzer
            .analyze(&audio, VuvAlgorithm::Energy)
            .await
            .unwrap();
        let stats = &analysis.statistics;

        assert_eq!(
            stats.total_frames,
            stats.voiced_frames + stats.unvoiced_frames
        );
        assert!(stats.voicing_rate >= 0.0 && stats.voicing_rate <= 1.0);
        assert!(stats.avg_voiced_probability >= 0.0 && stats.avg_voiced_probability <= 1.0);
        assert!(stats.longest_voiced_segment <= stats.total_frames);
    }
}
