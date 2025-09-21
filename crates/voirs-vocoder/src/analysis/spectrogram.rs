//! Spectrogram analysis and time-frequency decomposition

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2, s};
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

/// Comprehensive spectrogram analysis result
#[derive(Debug, Clone)]
pub struct SpectrogramAnalysis {
    /// Time-frequency magnitude matrix (time x frequency)
    pub magnitude_spectrogram: Array2<f32>,
    
    /// Time-frequency phase matrix (time x frequency)
    pub phase_spectrogram: Array2<f32>,
    
    /// Power spectrogram (magnitude squared)
    pub power_spectrogram: Array2<f32>,
    
    /// Log magnitude spectrogram (dB)
    pub log_magnitude_spectrogram: Array2<f32>,
    
    /// Time axis (seconds)
    pub time_axis: Vec<f32>,
    
    /// Frequency axis (Hz)
    pub frequency_axis: Vec<f32>,
    
    /// Temporal features
    pub temporal_features: TemporalFeatures,
    
    /// Spectrotemporal features
    pub spectrotemporal_features: SpectrotemporalFeatures,
}

/// Temporal features extracted from spectrogram
#[derive(Debug, Clone)]
pub struct TemporalFeatures {
    /// Onset detection function
    pub onset_strength: Vec<f32>,
    
    /// Detected onset times (seconds)
    pub onset_times: Vec<f32>,
    
    /// Tempo estimate (BPM)
    pub tempo: f32,
    
    /// Rhythmic regularity measure
    pub rhythmic_regularity: f32,
    
    /// Energy envelope
    pub energy_envelope: Vec<f32>,
    
    /// Temporal centroid (balance point in time)
    pub temporal_centroid: f32,
}

/// Spectrotemporal features
#[derive(Debug, Clone)]
pub struct SpectrotemporalFeatures {
    /// Spectral flux over time
    pub spectral_flux: Vec<f32>,
    
    /// Spectral rolloff over time
    pub spectral_rolloff: Vec<f32>,
    
    /// Spectral centroid over time
    pub spectral_centroid: Vec<f32>,
    
    /// Spectral bandwidth over time
    pub spectral_bandwidth: Vec<f32>,
    
    /// Harmonic-to-noise ratio over time
    pub hnr_over_time: Vec<f32>,
    
    /// Fundamental frequency tracking
    pub f0_contour: Vec<f32>,
    
    /// Formant tracking (first 3 formants)
    pub formant_contours: [Vec<f32>; 3],
}

impl Default for SpectrogramAnalysis {
    fn default() -> Self {
        Self {
            magnitude_spectrogram: Array2::zeros((0, 0)),
            phase_spectrogram: Array2::zeros((0, 0)),
            power_spectrogram: Array2::zeros((0, 0)),
            log_magnitude_spectrogram: Array2::zeros((0, 0)),
            time_axis: Vec::new(),
            frequency_axis: Vec::new(),
            temporal_features: TemporalFeatures::default(),
            spectrotemporal_features: SpectrotemporalFeatures::default(),
        }
    }
}

impl Default for TemporalFeatures {
    fn default() -> Self {
        Self {
            onset_strength: Vec::new(),
            onset_times: Vec::new(),
            tempo: 0.0,
            rhythmic_regularity: 0.0,
            energy_envelope: Vec::new(),
            temporal_centroid: 0.0,
        }
    }
}

impl Default for SpectrotemporalFeatures {
    fn default() -> Self {
        Self {
            spectral_flux: Vec::new(),
            spectral_rolloff: Vec::new(),
            spectral_centroid: Vec::new(),
            spectral_bandwidth: Vec::new(),
            hnr_over_time: Vec::new(),
            f0_contour: Vec::new(),
            formant_contours: [Vec::new(), Vec::new(), Vec::new()],
        }
    }
}

/// Advanced spectrogram analyzer
pub struct SpectrogramAnalyzer {
    sample_rate: u32,
    fft_planner: RealFftPlanner<f32>,
    fft_size: usize,
    hop_length: usize,
}

impl SpectrogramAnalyzer {
    /// Create new spectrogram analyzer
    pub fn new(sample_rate: u32, fft_size: usize, hop_length: usize) -> Self {
        Self {
            sample_rate,
            fft_planner: RealFftPlanner::new(),
            fft_size,
            hop_length,
        }
    }
    
    /// Compute comprehensive spectrogram analysis
    pub fn analyze(&mut self, samples: &Array1<f32>, window: &[f32]) -> Result<SpectrogramAnalysis> {
        if samples.is_empty() {
            return Ok(SpectrogramAnalysis::default());
        }
        
        // Compute basic spectrogram
        let (magnitude_spectrogram, phase_spectrogram) = self.compute_stft(samples, window)?;
        
        // Derive other representations
        let power_spectrogram = magnitude_spectrogram.mapv(|x| x * x);
        let log_magnitude_spectrogram = magnitude_spectrogram.mapv(|x| {
            if x > 1e-10 { 20.0 * x.log10() } else { -200.0 }
        });
        
        // Generate time and frequency axes
        let time_axis = self.generate_time_axis(magnitude_spectrogram.nrows());
        let frequency_axis = self.generate_frequency_axis();
        
        // Extract temporal features
        let temporal_features = self.extract_temporal_features(&magnitude_spectrogram, &time_axis);
        
        // Extract spectrotemporal features
        let spectrotemporal_features = self.extract_spectrotemporal_features(
            &magnitude_spectrogram,
            &frequency_axis,
            &time_axis
        );
        
        Ok(SpectrogramAnalysis {
            magnitude_spectrogram,
            phase_spectrogram,
            power_spectrogram,
            log_magnitude_spectrogram,
            time_axis,
            frequency_axis,
            temporal_features,
            spectrotemporal_features,
        })
    }
    
    /// Compute Short-Time Fourier Transform (STFT)
    fn compute_stft(&mut self, samples: &Array1<f32>, window: &[f32]) -> Result<(Array2<f32>, Array2<f32>)> {
        let n_frames = (samples.len().saturating_sub(self.fft_size)) / self.hop_length + 1;
        let n_bins = self.fft_size / 2 + 1;
        
        let mut magnitude_spec = Array2::zeros((n_frames, n_bins));
        let mut phase_spec = Array2::zeros((n_frames, n_bins));
        
        let mut fft = self.fft_planner.plan_fft_forward(self.fft_size);
        let mut input = vec![0.0; self.fft_size];
        let mut output = fft.make_output_vec();
        
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.fft_size).min(samples.len());
            
            // Clear input buffer
            input.fill(0.0);
            
            // Copy and window audio data
            for (i, &sample) in samples.slice(s![start..end]).iter().enumerate() {
                input[i] = sample * window[i];
            }
            
            // Compute FFT
            fft.process(&mut input, &mut output)
                .map_err(|_| VocoderError::ProcessingError("STFT computation failed".to_string()))?;
            
            // Extract magnitude and phase
            for (bin_idx, complex_val) in output.iter().enumerate() {
                magnitude_spec[[frame_idx, bin_idx]] = complex_val.norm();
                phase_spec[[frame_idx, bin_idx]] = complex_val.arg();
            }
        }
        
        Ok((magnitude_spec, phase_spec))
    }
    
    /// Generate time axis
    fn generate_time_axis(&self, n_frames: usize) -> Vec<f32> {
        (0..n_frames)
            .map(|i| (i * self.hop_length) as f32 / self.sample_rate as f32)
            .collect()
    }
    
    /// Generate frequency axis
    fn generate_frequency_axis(&self) -> Vec<f32> {
        let n_bins = self.fft_size / 2 + 1;
        (0..n_bins)
            .map(|i| i as f32 * self.sample_rate as f32 / self.fft_size as f32)
            .collect()
    }
    
    /// Extract temporal features from spectrogram
    fn extract_temporal_features(&self, magnitude_spec: &Array2<f32>, time_axis: &[f32]) -> TemporalFeatures {
        if magnitude_spec.is_empty() {
            return TemporalFeatures::default();
        }
        
        // Compute energy envelope (sum across frequency bins)
        let energy_envelope: Vec<f32> = magnitude_spec.rows()
            .into_iter()
            .map(|row| row.iter().map(|&x| x * x).sum())
            .collect();
        
        // Compute onset strength (spectral flux)
        let onset_strength = self.compute_onset_strength(magnitude_spec);
        
        // Detect onsets
        let onset_times = self.detect_onsets(&onset_strength, time_axis);
        
        // Estimate tempo
        let tempo = self.estimate_tempo(&onset_times);
        
        // Compute rhythmic regularity
        let rhythmic_regularity = self.compute_rhythmic_regularity(&onset_times);
        
        // Compute temporal centroid
        let temporal_centroid = self.compute_temporal_centroid(&energy_envelope, time_axis);
        
        TemporalFeatures {
            onset_strength,
            onset_times,
            tempo,
            rhythmic_regularity,
            energy_envelope,
            temporal_centroid,
        }
    }
    
    /// Extract spectrotemporal features
    fn extract_spectrotemporal_features(
        &self,
        magnitude_spec: &Array2<f32>,
        frequency_axis: &[f32],
        time_axis: &[f32],
    ) -> SpectrotemporalFeatures {
        if magnitude_spec.is_empty() {
            return SpectrotemporalFeatures::default();
        }
        
        let n_frames = magnitude_spec.nrows();
        let mut spectral_flux = Vec::with_capacity(n_frames);
        let mut spectral_rolloff = Vec::with_capacity(n_frames);
        let mut spectral_centroid = Vec::with_capacity(n_frames);
        let mut spectral_bandwidth = Vec::with_capacity(n_frames);
        let mut hnr_over_time = Vec::with_capacity(n_frames);
        let mut f0_contour = Vec::with_capacity(n_frames);
        
        let mut prev_magnitude: Option<Vec<f32>> = None;
        
        for frame_idx in 0..n_frames {
            let frame = magnitude_spec.row(frame_idx);
            let frame_vec: Vec<f32> = frame.to_vec();
            
            // Spectral flux
            if let Some(ref prev) = prev_magnitude {
                let flux = self.compute_spectral_flux(prev, &frame_vec);
                spectral_flux.push(flux);
            } else {
                spectral_flux.push(0.0);
            }
            
            // Spectral features for this frame
            let centroid = self.compute_frame_spectral_centroid(frequency_axis, &frame_vec);
            let rolloff = self.compute_frame_spectral_rolloff(frequency_axis, &frame_vec, 0.85);
            let bandwidth = self.compute_frame_spectral_bandwidth(frequency_axis, &frame_vec, centroid);
            
            spectral_centroid.push(centroid);
            spectral_rolloff.push(rolloff);
            spectral_bandwidth.push(bandwidth);
            
            // Harmonic-to-noise ratio
            let hnr = self.compute_frame_hnr(&frame_vec);
            hnr_over_time.push(hnr);
            
            // Fundamental frequency estimation
            let f0 = self.estimate_frame_f0(frequency_axis, &frame_vec);
            f0_contour.push(f0);
            
            prev_magnitude = Some(frame_vec);
        }
        
        // Formant tracking (simplified)
        let formant_contours = self.track_formants(magnitude_spec, frequency_axis);
        
        SpectrotemporalFeatures {
            spectral_flux,
            spectral_rolloff,
            spectral_centroid,
            spectral_bandwidth,
            hnr_over_time,
            f0_contour,
            formant_contours,
        }
    }
    
    /// Compute onset strength using spectral flux
    fn compute_onset_strength(&self, magnitude_spec: &Array2<f32>) -> Vec<f32> {
        let n_frames = magnitude_spec.nrows();
        let mut onset_strength = vec![0.0; n_frames];
        
        for frame_idx in 1..n_frames {
            let current_frame = magnitude_spec.row(frame_idx);
            let prev_frame = magnitude_spec.row(frame_idx - 1);
            
            let flux: f32 = current_frame.iter()
                .zip(prev_frame.iter())
                .map(|(&curr, &prev)| (curr - prev).max(0.0))
                .sum();
            
            onset_strength[frame_idx] = flux;
        }
        
        onset_strength
    }
    
    /// Detect onsets using peak picking
    fn detect_onsets(&self, onset_strength: &[f32], time_axis: &[f32]) -> Vec<f32> {
        let mut onsets = Vec::new();
        
        if onset_strength.len() < 3 {
            return onsets;
        }
        
        // Simple peak picking with minimum threshold
        let mean_strength: f32 = onset_strength.iter().sum::<f32>() / onset_strength.len() as f32;
        let threshold = mean_strength * 1.5;
        
        for i in 1..onset_strength.len()-1 {
            if onset_strength[i] > threshold &&
               onset_strength[i] > onset_strength[i-1] &&
               onset_strength[i] > onset_strength[i+1] {
                if let Some(&time) = time_axis.get(i) {
                    onsets.push(time);
                }
            }
        }
        
        onsets
    }
    
    /// Estimate tempo from onset times
    fn estimate_tempo(&self, onset_times: &[f32]) -> f32 {
        if onset_times.len() < 2 {
            return 0.0;
        }
        
        // Compute inter-onset intervals
        let intervals: Vec<f32> = onset_times.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        if intervals.is_empty() {
            return 0.0;
        }
        
        // Find median interval
        let mut sorted_intervals = intervals.clone();
        sorted_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_interval = sorted_intervals[sorted_intervals.len() / 2];
        
        // Convert to BPM
        if median_interval > 0.0 {
            60.0 / median_interval
        } else {
            0.0
        }
    }
    
    /// Compute rhythmic regularity
    fn compute_rhythmic_regularity(&self, onset_times: &[f32]) -> f32 {
        if onset_times.len() < 3 {
            return 0.0;
        }
        
        // Compute coefficient of variation of inter-onset intervals
        let intervals: Vec<f32> = onset_times.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        if intervals.is_empty() {
            return 0.0;
        }
        
        let mean: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance: f32 = intervals.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / intervals.len() as f32;
        
        let std_dev = variance.sqrt();
        
        if mean > 1e-6 {
            1.0 - (std_dev / mean).min(1.0) // Higher regularity = lower coefficient of variation
        } else {
            0.0
        }
    }
    
    /// Compute temporal centroid
    fn compute_temporal_centroid(&self, energy_envelope: &[f32], time_axis: &[f32]) -> f32 {
        if energy_envelope.is_empty() || time_axis.is_empty() {
            return 0.0;
        }
        
        let total_energy: f32 = energy_envelope.iter().sum();
        if total_energy <= 1e-10 {
            return 0.0;
        }
        
        let weighted_sum: f32 = energy_envelope.iter()
            .zip(time_axis.iter())
            .map(|(&energy, &time)| energy * time)
            .sum();
        
        weighted_sum / total_energy
    }
    
    /// Compute spectral flux between two frames
    fn compute_spectral_flux(&self, prev_frame: &[f32], curr_frame: &[f32]) -> f32 {
        prev_frame.iter()
            .zip(curr_frame.iter())
            .map(|(&prev, &curr)| (curr - prev).max(0.0))
            .sum()
    }
    
    /// Compute spectral centroid for a single frame
    fn compute_frame_spectral_centroid(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
        let weighted_sum: f32 = frequencies.iter()
            .zip(magnitudes.iter())
            .map(|(&freq, &mag)| freq * mag)
            .sum();
        
        let magnitude_sum: f32 = magnitudes.iter().sum();
        
        if magnitude_sum > 1e-10 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
    
    /// Compute spectral rolloff for a single frame
    fn compute_frame_spectral_rolloff(&self, frequencies: &[f32], magnitudes: &[f32], threshold: f32) -> f32 {
        let total_energy: f32 = magnitudes.iter().map(|&m| m * m).sum();
        let target_energy = total_energy * threshold;
        
        let mut cumulative_energy = 0.0;
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            cumulative_energy += mag * mag;
            if cumulative_energy >= target_energy {
                return *freq;
            }
        }
        
        frequencies.last().copied().unwrap_or(0.0)
    }
    
    /// Compute spectral bandwidth for a single frame
    fn compute_frame_spectral_bandwidth(&self, frequencies: &[f32], magnitudes: &[f32], centroid: f32) -> f32 {
        let mut weighted_variance = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            let diff = freq - centroid;
            weighted_variance += diff * diff * mag;
            magnitude_sum += mag;
        }
        
        if magnitude_sum > 1e-10 {
            (weighted_variance / magnitude_sum).sqrt()
        } else {
            0.0
        }
    }
    
    /// Compute harmonic-to-noise ratio for a single frame
    fn compute_frame_hnr(&self, magnitudes: &[f32]) -> f32 {
        if magnitudes.len() < 4 {
            return 0.0;
        }
        
        // Simplified HNR: ratio of harmonic peaks to overall energy
        let peak_indices = self.find_spectral_peaks(magnitudes);
        let harmonic_energy: f32 = peak_indices.iter()
            .map(|&idx| magnitudes[idx] * magnitudes[idx])
            .sum();
        
        let total_energy: f32 = magnitudes.iter().map(|&m| m * m).sum();
        
        if total_energy > 1e-10 && harmonic_energy > 1e-10 {
            10.0 * (harmonic_energy / (total_energy - harmonic_energy)).log10()
        } else {
            -20.0 // Low HNR
        }
    }
    
    /// Find spectral peaks
    fn find_spectral_peaks(&self, magnitudes: &[f32]) -> Vec<usize> {
        let mut peaks = Vec::new();
        
        if magnitudes.len() < 3 {
            return peaks;
        }
        
        for i in 1..magnitudes.len()-1 {
            if magnitudes[i] > magnitudes[i-1] && magnitudes[i] > magnitudes[i+1] {
                peaks.push(i);
            }
        }
        
        peaks
    }
    
    /// Estimate fundamental frequency for a single frame
    fn estimate_frame_f0(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
        // Simple F0 estimation: find the strongest low-frequency peak
        let mut max_magnitude = 0.0;
        let mut f0 = 0.0;
        
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            // Look for F0 in typical speech range (80-400 Hz)
            if *freq >= 80.0 && *freq <= 400.0 && *mag > max_magnitude {
                max_magnitude = *mag;
                f0 = *freq;
            }
        }
        
        f0
    }
    
    /// Track formants (simplified implementation)
    fn track_formants(&self, magnitude_spec: &Array2<f32>, frequencies: &[f32]) -> [Vec<f32>; 3] {
        let n_frames = magnitude_spec.nrows();
        let mut formant1 = Vec::with_capacity(n_frames);
        let mut formant2 = Vec::with_capacity(n_frames);
        let mut formant3 = Vec::with_capacity(n_frames);
        
        for frame_idx in 0..n_frames {
            let frame = magnitude_spec.row(frame_idx);
            let peaks = self.find_spectral_peaks(&frame.to_vec());
            
            // Extract formant frequencies (simplified)
            let mut formant_freqs = Vec::new();
            for &peak_idx in &peaks {
                if let Some(&freq) = frequencies.get(peak_idx) {
                    if freq >= 200.0 && freq <= 3500.0 { // Typical formant range
                        formant_freqs.push(freq);
                    }
                }
            }
            
            formant_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            formant1.push(formant_freqs.get(0).copied().unwrap_or(0.0));
            formant2.push(formant_freqs.get(1).copied().unwrap_or(0.0));
            formant3.push(formant_freqs.get(2).copied().unwrap_or(0.0));
        }
        
        [formant1, formant2, formant3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_spectrogram_analyzer_creation() {
        let analyzer = SpectrogramAnalyzer::new(22050, 1024, 256);
        assert_eq!(analyzer.sample_rate, 22050);
        assert_eq!(analyzer.fft_size, 1024);
        assert_eq!(analyzer.hop_length, 256);
    }
    
    #[test]
    fn test_time_axis_generation() {
        let analyzer = SpectrogramAnalyzer::new(22050, 1024, 256);
        let time_axis = analyzer.generate_time_axis(10);
        
        assert_eq!(time_axis.len(), 10);
        assert!((time_axis[0] - 0.0).abs() < 1e-6);
        assert!((time_axis[1] - (256.0 / 22050.0)).abs() < 1e-6);
    }
    
    #[test]
    fn test_frequency_axis_generation() {
        let analyzer = SpectrogramAnalyzer::new(22050, 1024, 256);
        let freq_axis = analyzer.generate_frequency_axis();
        
        assert_eq!(freq_axis.len(), 513); // (1024/2) + 1
        assert!((freq_axis[0] - 0.0).abs() < 1e-6);
        assert!((freq_axis.last().unwrap() - 11025.0).abs() < 1e-3); // Nyquist frequency
    }
    
    #[test]
    fn test_onset_detection() {
        let analyzer = SpectrogramAnalyzer::new(22050, 1024, 256);
        
        // Create a simple onset strength function with peaks
        let onset_strength = vec![0.1, 0.2, 1.0, 0.3, 0.1, 0.2, 1.5, 0.2, 0.1];
        let time_axis: Vec<f32> = (0..onset_strength.len()).map(|i| i as f32 * 0.1).collect();
        
        let onsets = analyzer.detect_onsets(&onset_strength, &time_axis);
        
        // Should detect peaks at indices 2 and 6
        assert!(!onsets.is_empty());
    }
    
    #[test]
    fn test_tempo_estimation() {
        let analyzer = SpectrogramAnalyzer::new(22050, 1024, 256);
        
        // Regular onsets at 120 BPM (0.5 second intervals)
        let onset_times = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let tempo = analyzer.estimate_tempo(&onset_times);
        
        // Should be around 120 BPM
        assert!((tempo - 120.0).abs() < 10.0);
    }
    
    #[test]
    fn test_spectral_centroid_computation() {
        let analyzer = SpectrogramAnalyzer::new(22050, 1024, 256);
        
        let frequencies = vec![100.0, 200.0, 300.0, 400.0];
        let magnitudes = vec![0.1, 1.0, 0.1, 0.1]; // Peak at 200 Hz
        
        let centroid = analyzer.compute_frame_spectral_centroid(&frequencies, &magnitudes);
        
        // Should be close to 200 Hz
        assert!((centroid - 200.0).abs() < 50.0);
    }
}