//! Artifact reduction processor for singing voice vocoder.

use crate::models::singing::config::ArtifactReductionConfig;
use anyhow::Result;
#[cfg(test)]
use scirs2_core::ndarray::Array1;
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::Complex;
use scirs2_fft::{FftPlanner, RealFftPlanner};
use std::collections::VecDeque;

/// Processor for artifact reduction in singing voice vocoding
pub struct ArtifactReductionProcessor {
    /// Configuration
    config: ArtifactReductionConfig,
    /// Window size for analysis
    window_size: usize,
    /// Hop size for analysis
    hop_size: usize,
    /// Sample rate
    sample_rate: u32,
    /// Spectral history for temporal smoothing
    spectral_history: VecDeque<Vec<f32>>,
    /// Noise floor estimation
    noise_floor: Vec<f32>,
    /// Frame counter for adaptive processing
    frame_counter: usize,
}

/// Artifact analysis result
#[derive(Debug, Clone)]
pub struct ArtifactAnalysis {
    /// Spectral noise level
    pub noise_level: f32,
    /// Harmonic artifacts detected
    pub harmonic_artifacts: Vec<HarmonicArtifact>,
    /// Temporal artifacts detected
    pub temporal_artifacts: Vec<TemporalArtifact>,
    /// Overall artifact score
    pub artifact_score: f32,
}

/// Harmonic artifact information
#[derive(Debug, Clone)]
pub struct HarmonicArtifact {
    /// Frequency of the artifact
    pub frequency: f32,
    /// Magnitude of the artifact
    pub magnitude: f32,
    /// Artifact type
    pub artifact_type: HarmonicArtifactType,
}

/// Temporal artifact information
#[derive(Debug, Clone)]
pub struct TemporalArtifact {
    /// Time position of the artifact
    pub time_position: f32,
    /// Duration of the artifact
    pub duration: f32,
    /// Artifact type
    pub artifact_type: TemporalArtifactType,
}

/// Types of harmonic artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmonicArtifactType {
    /// Aliasing artifacts
    Aliasing,
    /// Quantization noise
    Quantization,
    /// Harmonic distortion
    Distortion,
    /// Intermodulation artifacts
    Intermodulation,
}

/// Types of temporal artifacts
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalArtifactType {
    /// Discontinuities
    Discontinuity,
    /// Clicks and pops
    Click,
    /// Warbling
    Warble,
    /// Phasiness
    Phase,
}

impl ArtifactReductionProcessor {
    /// Create new artifact reduction processor
    pub fn new(config: &ArtifactReductionConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            window_size: 2048,
            hop_size: 512,
            sample_rate: 22050,
            spectral_history: VecDeque::with_capacity(10),
            noise_floor: vec![0.0; 1024],
            frame_counter: 0,
        })
    }

    /// Process mel spectrogram for artifact reduction
    pub fn process(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.config.enable_reduction {
            return Ok(mel_spectrogram.clone());
        }

        let mut processed = mel_spectrogram.clone();
        let frames = mel_spectrogram.shape()[1];

        // Process each frame
        for frame_idx in 0..frames {
            let frame = mel_spectrogram.column(frame_idx);
            let analysis = self.analyze_artifacts(&frame)?;

            // Apply artifact reduction
            self.apply_artifact_reduction(&mut processed, frame_idx, &analysis)?;

            // Update frame counter
            self.frame_counter += 1;
        }

        Ok(processed)
    }

    /// Analyze artifacts in a frame
    fn analyze_artifacts(&mut self, frame: &ArrayView1<f32>) -> Result<ArtifactAnalysis> {
        // Convert mel frame to linear spectrum
        let spectrum = self.mel_to_linear_spectrum(frame)?;

        // Apply windowing
        let windowed_spectrum = self.apply_hann_window(&spectrum);

        // Perform FFT
        let fft_input: Vec<Complex<f32>> = windowed_spectrum
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        let fft_output_f64 = scirs2_fft::fft(&fft_input, None)?;

        // Convert f64 output to f32 and extract magnitude spectrum
        let magnitude_spectrum: Vec<f32> =
            fft_output_f64.iter().map(|c| (c.norm()) as f32).collect();

        // Update spectral history
        self.spectral_history.push_back(magnitude_spectrum.clone());
        if self.spectral_history.len() > 10 {
            self.spectral_history.pop_front();
        }

        // Update noise floor estimation
        self.update_noise_floor(&magnitude_spectrum)?;

        // Analyze different types of artifacts
        let noise_level = self.calculate_noise_level(&magnitude_spectrum)?;
        let harmonic_artifacts = self.detect_harmonic_artifacts(&magnitude_spectrum)?;
        let temporal_artifacts = self.detect_temporal_artifacts()?;
        let artifact_score =
            self.calculate_artifact_score(&harmonic_artifacts, &temporal_artifacts)?;

        Ok(ArtifactAnalysis {
            noise_level,
            harmonic_artifacts,
            temporal_artifacts,
            artifact_score,
        })
    }

    /// Update noise floor estimation
    fn update_noise_floor(&mut self, spectrum: &[f32]) -> Result<()> {
        let alpha = 0.1; // Smoothing factor

        for (i, &magnitude) in spectrum.iter().enumerate() {
            if i < self.noise_floor.len() {
                // Use minimum statistics for noise floor estimation
                if magnitude < self.noise_floor[i] || self.frame_counter < 10 {
                    self.noise_floor[i] = magnitude;
                } else {
                    self.noise_floor[i] = (1.0 - alpha) * self.noise_floor[i] + alpha * magnitude;
                }
            }
        }

        Ok(())
    }

    /// Calculate noise level
    fn calculate_noise_level(&mut self, spectrum: &[f32]) -> Result<f32> {
        let mut noise_energy = 0.0;
        let mut signal_energy = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let noise_floor = if i < self.noise_floor.len() {
                self.noise_floor[i]
            } else {
                0.0
            };

            noise_energy += noise_floor * noise_floor;
            signal_energy += magnitude * magnitude;
        }

        let noise_level = if signal_energy > 0.0 {
            (noise_energy / signal_energy).sqrt()
        } else {
            0.0
        };

        Ok(noise_level.clamp(0.0, 1.0))
    }

    /// Detect harmonic artifacts
    fn detect_harmonic_artifacts(&self, spectrum: &[f32]) -> Result<Vec<HarmonicArtifact>> {
        let mut artifacts = Vec::new();
        let freq_range = &self.config.artifact_frequency_range;

        let min_bin =
            (freq_range.0 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let max_bin =
            (freq_range.1 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        // Detect spectral peaks that might be artifacts
        for bin in min_bin..max_bin.min(spectrum.len()) {
            let frequency = (bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
            let magnitude = spectrum[bin];

            // Check for aliasing artifacts (near Nyquist frequency)
            if frequency > self.sample_rate as f32 * 0.4 {
                let noise_threshold = if bin < self.noise_floor.len() {
                    self.noise_floor[bin] * 5.0
                } else {
                    0.1
                };

                if magnitude > noise_threshold {
                    artifacts.push(HarmonicArtifact {
                        frequency,
                        magnitude,
                        artifact_type: HarmonicArtifactType::Aliasing,
                    });
                }
            }

            // Check for quantization noise (spiky spectrum)
            if bin > 0 && bin < spectrum.len() - 1 {
                let neighbors_avg = (spectrum[bin - 1] + spectrum[bin + 1]) / 2.0;
                let peak_ratio = magnitude / neighbors_avg;

                if peak_ratio > 3.0 && magnitude > 0.1 {
                    artifacts.push(HarmonicArtifact {
                        frequency,
                        magnitude,
                        artifact_type: HarmonicArtifactType::Quantization,
                    });
                }
            }

            // Check for harmonic distortion (unexpected harmonics)
            if self.is_unexpected_harmonic(frequency, magnitude)? {
                artifacts.push(HarmonicArtifact {
                    frequency,
                    magnitude,
                    artifact_type: HarmonicArtifactType::Distortion,
                });
            }
        }

        Ok(artifacts)
    }

    /// Check if a frequency is an unexpected harmonic
    fn is_unexpected_harmonic(&self, frequency: f32, magnitude: f32) -> Result<bool> {
        // Simple heuristic: check if frequency is a high-order harmonic with unexpected magnitude
        let fundamental_candidates = [110.0, 220.0, 440.0, 880.0]; // Common fundamental frequencies

        for &fundamental in &fundamental_candidates {
            let harmonic_number = (frequency / fundamental).round() as i32;
            if harmonic_number > 1 && harmonic_number < 20 {
                let expected_magnitude = 1.0 / (harmonic_number as f32).sqrt();
                if magnitude > expected_magnitude * 3.0 {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Detect temporal artifacts
    fn detect_temporal_artifacts(&self) -> Result<Vec<TemporalArtifact>> {
        let mut artifacts = Vec::new();

        if self.spectral_history.len() < 3 {
            return Ok(artifacts);
        }

        // Get current and previous frames
        let current_frame = self.spectral_history.back().unwrap();
        let prev_frame = &self.spectral_history[self.spectral_history.len() - 2];

        // Detect discontinuities
        let discontinuity_score =
            self.calculate_spectral_discontinuity(prev_frame, current_frame)?;
        if discontinuity_score > 0.5 {
            artifacts.push(TemporalArtifact {
                time_position: self.frame_counter as f32 * self.hop_size as f32
                    / self.sample_rate as f32,
                duration: self.hop_size as f32 / self.sample_rate as f32,
                artifact_type: TemporalArtifactType::Discontinuity,
            });
        }

        // Detect clicks and pops
        let click_score = self.calculate_click_score(prev_frame, current_frame)?;
        if click_score > 0.3 {
            artifacts.push(TemporalArtifact {
                time_position: self.frame_counter as f32 * self.hop_size as f32
                    / self.sample_rate as f32,
                duration: self.hop_size as f32 / self.sample_rate as f32,
                artifact_type: TemporalArtifactType::Click,
            });
        }

        // Detect warbling (if we have enough history)
        if self.spectral_history.len() >= 5 {
            let warble_score = self.calculate_warble_score()?;
            if warble_score > 0.4 {
                artifacts.push(TemporalArtifact {
                    time_position: self.frame_counter as f32 * self.hop_size as f32
                        / self.sample_rate as f32,
                    duration: 5.0 * self.hop_size as f32 / self.sample_rate as f32,
                    artifact_type: TemporalArtifactType::Warble,
                });
            }
        }

        Ok(artifacts)
    }

    /// Calculate spectral discontinuity score
    fn calculate_spectral_discontinuity(
        &self,
        prev_frame: &[f32],
        current_frame: &[f32],
    ) -> Result<f32> {
        let mut diff_sum = 0.0;
        let mut total_energy = 0.0;

        let min_len = prev_frame.len().min(current_frame.len());

        for i in 0..min_len {
            let diff = (current_frame[i] - prev_frame[i]).abs();
            diff_sum += diff;
            total_energy += prev_frame[i] + current_frame[i];
        }

        let discontinuity_score = if total_energy > 0.0 {
            diff_sum / total_energy
        } else {
            0.0
        };

        Ok(discontinuity_score.clamp(0.0, 1.0))
    }

    /// Calculate click score
    fn calculate_click_score(&self, prev_frame: &[f32], current_frame: &[f32]) -> Result<f32> {
        let mut high_freq_diff = 0.0;
        let mut high_freq_energy = 0.0;

        let min_len = prev_frame.len().min(current_frame.len());
        let high_freq_start = min_len * 3 / 4; // Focus on high frequencies

        for i in high_freq_start..min_len {
            let diff = (current_frame[i] - prev_frame[i]).abs();
            high_freq_diff += diff * diff;
            high_freq_energy += prev_frame[i] * prev_frame[i] + current_frame[i] * current_frame[i];
        }

        let click_score = if high_freq_energy > 0.0 {
            (high_freq_diff / high_freq_energy).sqrt()
        } else {
            0.0
        };

        Ok(click_score.clamp(0.0, 1.0))
    }

    /// Calculate warble score
    fn calculate_warble_score(&self) -> Result<f32> {
        if self.spectral_history.len() < 5 {
            return Ok(0.0);
        }

        let _frames = self.spectral_history.len();
        let mut centroid_variance = 0.0;
        let mut centroids = Vec::new();

        // Calculate spectral centroid for each frame
        for frame in &self.spectral_history {
            let centroid = self.calculate_spectral_centroid(frame)?;
            centroids.push(centroid);
        }

        // Calculate variance of centroids
        let mean_centroid = centroids.iter().sum::<f32>() / centroids.len() as f32;
        for &centroid in &centroids {
            centroid_variance += (centroid - mean_centroid).powi(2);
        }
        centroid_variance /= centroids.len() as f32;

        // Normalize warble score
        let warble_score = (centroid_variance / 1000.0).sqrt().clamp(0.0, 1.0);

        Ok(warble_score)
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, spectrum: &[f32]) -> Result<f32> {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (bin, &magnitude) in spectrum.iter().enumerate() {
            let frequency = (bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        let centroid = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };

        Ok(centroid)
    }

    /// Calculate overall artifact score
    fn calculate_artifact_score(
        &self,
        harmonic_artifacts: &[HarmonicArtifact],
        temporal_artifacts: &[TemporalArtifact],
    ) -> Result<f32> {
        let harmonic_score = harmonic_artifacts.iter().map(|a| a.magnitude).sum::<f32>()
            / harmonic_artifacts.len().max(1) as f32;

        let temporal_score = temporal_artifacts.len() as f32 / 10.0; // Normalize by expected max

        let overall_score = (harmonic_score * 0.6 + temporal_score * 0.4).clamp(0.0, 1.0);

        Ok(overall_score)
    }

    /// Apply artifact reduction to frame
    fn apply_artifact_reduction(
        &self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        analysis: &ArtifactAnalysis,
    ) -> Result<()> {
        let _mel_bins = mel_spectrogram.shape()[0];

        // Apply spectral noise reduction
        if analysis.noise_level > 0.1 {
            self.apply_spectral_noise_reduction(mel_spectrogram, frame_idx, analysis.noise_level)?;
        }

        // Apply harmonic artifact reduction
        for artifact in &analysis.harmonic_artifacts {
            self.apply_harmonic_artifact_reduction(mel_spectrogram, frame_idx, artifact)?;
        }

        // Apply temporal artifact reduction
        if !analysis.temporal_artifacts.is_empty() {
            self.apply_temporal_artifact_reduction(
                mel_spectrogram,
                frame_idx,
                &analysis.temporal_artifacts,
            )?;
        }

        Ok(())
    }

    /// Apply spectral noise reduction
    fn apply_spectral_noise_reduction(
        &self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        noise_level: f32,
    ) -> Result<()> {
        let mel_bins = mel_spectrogram.shape()[0];
        let reduction_factor = 1.0 - (noise_level * self.config.noise_reduction_strength);

        for bin_idx in 0..mel_bins {
            let current_value = mel_spectrogram[[bin_idx, frame_idx]];
            mel_spectrogram[[bin_idx, frame_idx]] = current_value * reduction_factor;
        }

        Ok(())
    }

    /// Apply harmonic artifact reduction
    fn apply_harmonic_artifact_reduction(
        &self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        artifact: &HarmonicArtifact,
    ) -> Result<()> {
        let mel_bins = mel_spectrogram.shape()[0];
        let artifact_mel_bin = self.hz_to_mel_bin(artifact.frequency, mel_bins);

        let reduction_strength = match artifact.artifact_type {
            HarmonicArtifactType::Aliasing => self.config.harmonic_artifact_reduction * 1.2,
            HarmonicArtifactType::Quantization => self.config.harmonic_artifact_reduction * 0.8,
            HarmonicArtifactType::Distortion => self.config.harmonic_artifact_reduction,
            HarmonicArtifactType::Intermodulation => self.config.harmonic_artifact_reduction * 0.9,
        };

        let _reduction_factor = 1.0 - (artifact.magnitude * reduction_strength);

        // Apply reduction to artifact frequency and neighboring bins
        for offset in -2..=2 {
            let bin_idx = (artifact_mel_bin as i32 + offset) as usize;
            if bin_idx < mel_bins {
                let current_value = mel_spectrogram[[bin_idx, frame_idx]];
                let weight = 1.0 - (offset.abs() as f32 / 2.0);
                let local_reduction = 1.0 - (reduction_strength * weight);
                mel_spectrogram[[bin_idx, frame_idx]] = current_value * local_reduction;
            }
        }

        Ok(())
    }

    /// Apply temporal artifact reduction
    fn apply_temporal_artifact_reduction(
        &self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        _artifacts: &[TemporalArtifact],
    ) -> Result<()> {
        let mel_bins = mel_spectrogram.shape()[0];
        let reduction_strength = self.config.temporal_artifact_reduction;

        // Apply smoothing for temporal artifacts
        for bin_idx in 0..mel_bins {
            let current_value = mel_spectrogram[[bin_idx, frame_idx]];

            // Simple temporal smoothing
            let smoothed_value = if frame_idx > 0 {
                let prev_value = mel_spectrogram[[bin_idx, frame_idx - 1]];
                current_value * (1.0 - reduction_strength) + prev_value * reduction_strength
            } else {
                current_value
            };

            mel_spectrogram[[bin_idx, frame_idx]] = smoothed_value;
        }

        Ok(())
    }

    /// Convert mel frame to linear spectrum
    fn mel_to_linear_spectrum(&self, frame: &ArrayView1<f32>) -> Result<Vec<f32>> {
        let mel_bins = frame.len();
        let mut spectrum = vec![0.0; self.window_size / 2 + 1];

        for (mel_idx, &mel_value) in frame.iter().enumerate() {
            let mel_freq =
                (mel_idx as f32 / mel_bins as f32) * self.hz_to_mel(self.sample_rate as f32 / 2.0);
            let hz_freq = self.mel_to_hz(mel_freq);
            let spec_bin =
                (hz_freq / (self.sample_rate as f32 / 2.0) * spectrum.len() as f32) as usize;

            if spec_bin < spectrum.len() {
                spectrum[spec_bin] = mel_value;
            }
        }

        Ok(spectrum)
    }

    /// Apply Hann window to spectrum
    fn apply_hann_window(&self, spectrum: &[f32]) -> Vec<f32> {
        let n = spectrum.len();
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let window =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
                value * window
            })
            .collect()
    }

    /// Convert mel frequency to Hz
    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (mel / 1127.0).exp() - 700.0
    }

    /// Convert Hz to mel frequency
    fn hz_to_mel(&self, hz: f32) -> f32 {
        1127.0 * (1.0 + hz / 700.0).ln()
    }

    /// Convert Hz to mel bin index
    fn hz_to_mel_bin(&self, hz: f32, mel_bins: usize) -> usize {
        let mel_freq = self.hz_to_mel(hz);
        let max_mel = self.hz_to_mel(self.sample_rate as f32 / 2.0);
        let bin_idx = (mel_freq / max_mel * mel_bins as f32) as usize;
        bin_idx.min(mel_bins - 1)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &ArtifactReductionConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }

    /// Get artifact statistics
    pub fn get_artifact_stats(&self, analysis: &ArtifactAnalysis) -> ArtifactStats {
        ArtifactStats {
            noise_level: analysis.noise_level,
            harmonic_artifact_count: analysis.harmonic_artifacts.len() as u32,
            temporal_artifact_count: analysis.temporal_artifacts.len() as u32,
            overall_artifact_score: analysis.artifact_score,
            dominant_artifact_type: self.get_dominant_artifact_type(&analysis.harmonic_artifacts),
        }
    }

    /// Get dominant artifact type
    fn get_dominant_artifact_type(&self, artifacts: &[HarmonicArtifact]) -> HarmonicArtifactType {
        let mut type_counts = std::collections::HashMap::new();

        for artifact in artifacts {
            *type_counts.entry(artifact.artifact_type).or_insert(0) += 1;
        }

        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(artifact_type, _)| artifact_type)
            .unwrap_or(HarmonicArtifactType::Quantization)
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.spectral_history.clear();
        self.noise_floor.fill(0.0);
        self.frame_counter = 0;
    }
}

/// Statistics for artifact analysis
#[derive(Debug, Clone)]
pub struct ArtifactStats {
    /// Noise level
    pub noise_level: f32,
    /// Number of harmonic artifacts
    pub harmonic_artifact_count: u32,
    /// Number of temporal artifacts
    pub temporal_artifact_count: u32,
    /// Overall artifact score
    pub overall_artifact_score: f32,
    /// Dominant artifact type
    pub dominant_artifact_type: HarmonicArtifactType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_reduction_processor_creation() {
        let config = ArtifactReductionConfig::default();
        let processor = ArtifactReductionProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_artifact_analysis() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        // Create sample mel frame
        let frame = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.3, 0.1]);
        let frame_view = frame.view();

        let analysis = processor.analyze_artifacts(&frame_view);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_noise_level_calculation() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        let spectrum = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let noise_level = processor.calculate_noise_level(&spectrum);
        assert!(noise_level.is_ok());
        let noise_val = noise_level.unwrap();
        assert!((0.0..=1.0).contains(&noise_val));
    }

    #[test]
    fn test_harmonic_artifact_detection() {
        let config = ArtifactReductionConfig::default();
        let processor = ArtifactReductionProcessor::new(&config).unwrap();

        // Create spectrum with potential artifacts
        let spectrum = vec![0.1; 1024];

        let artifacts = processor.detect_harmonic_artifacts(&spectrum);
        assert!(artifacts.is_ok());
    }

    #[test]
    fn test_temporal_artifact_detection() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        // Add some spectral history
        processor.spectral_history.push_back(vec![0.1, 0.2, 0.3]);
        processor.spectral_history.push_back(vec![0.2, 0.3, 0.4]);
        processor.spectral_history.push_back(vec![0.8, 0.9, 1.0]); // Sudden change

        let artifacts = processor.detect_temporal_artifacts();
        assert!(artifacts.is_ok());
    }

    #[test]
    fn test_spectral_discontinuity() {
        let config = ArtifactReductionConfig::default();
        let processor = ArtifactReductionProcessor::new(&config).unwrap();

        let prev_frame = vec![0.1, 0.2, 0.3];
        let current_frame = vec![0.8, 0.9, 1.0];

        let discontinuity = processor.calculate_spectral_discontinuity(&prev_frame, &current_frame);
        assert!(discontinuity.is_ok());
        assert!(discontinuity.unwrap() > 0.0);
    }

    #[test]
    fn test_click_score() {
        let config = ArtifactReductionConfig::default();
        let processor = ArtifactReductionProcessor::new(&config).unwrap();

        let prev_frame = vec![0.1, 0.1, 0.1, 0.1];
        let current_frame = vec![0.1, 0.1, 0.9, 0.9]; // High-frequency spike

        let click_score = processor.calculate_click_score(&prev_frame, &current_frame);
        assert!(click_score.is_ok());
        assert!(click_score.unwrap() > 0.0);
    }

    #[test]
    fn test_artifact_reduction_processing() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = processor.process(&mel);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.shape(), mel.shape());
    }

    #[test]
    fn test_noise_floor_update() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        let spectrum = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result = processor.update_noise_floor(&spectrum);
        assert!(result.is_ok());
        assert!(processor.noise_floor.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_spectral_noise_reduction() {
        let config = ArtifactReductionConfig::default();
        let processor = ArtifactReductionProcessor::new(&config).unwrap();

        let mut mel = Array2::ones((80, 100));
        let original_value = mel[[40, 50]];

        let result = processor.apply_spectral_noise_reduction(&mut mel, 50, 0.5);
        assert!(result.is_ok());

        // Value should be reduced
        assert!(mel[[40, 50]] < original_value);
    }

    #[test]
    fn test_config_update() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        let new_config = ArtifactReductionConfig {
            noise_reduction_strength: 0.8,
            ..Default::default()
        };

        let result = processor.update_config(&new_config);
        assert!(result.is_ok());
        assert_eq!(processor.config.noise_reduction_strength, 0.8);
    }

    #[test]
    fn test_artifact_stats() {
        let config = ArtifactReductionConfig::default();
        let processor = ArtifactReductionProcessor::new(&config).unwrap();

        let analysis = ArtifactAnalysis {
            noise_level: 0.3,
            harmonic_artifacts: vec![HarmonicArtifact {
                frequency: 1000.0,
                magnitude: 0.5,
                artifact_type: HarmonicArtifactType::Aliasing,
            }],
            temporal_artifacts: vec![],
            artifact_score: 0.4,
        };

        let stats = processor.get_artifact_stats(&analysis);
        assert_eq!(stats.noise_level, 0.3);
        assert_eq!(stats.harmonic_artifact_count, 1);
        assert_eq!(stats.temporal_artifact_count, 0);
        assert_eq!(stats.overall_artifact_score, 0.4);
    }

    #[test]
    fn test_processor_reset() {
        let config = ArtifactReductionConfig::default();
        let mut processor = ArtifactReductionProcessor::new(&config).unwrap();

        processor.spectral_history.push_back(vec![0.1, 0.2, 0.3]);
        processor.frame_counter = 10;
        processor.noise_floor[0] = 0.5;

        processor.reset();

        assert!(processor.spectral_history.is_empty());
        assert_eq!(processor.frame_counter, 0);
        assert_eq!(processor.noise_floor[0], 0.0);
    }
}
