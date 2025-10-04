//! Advanced audio analysis tools
//!
//! This module provides comprehensive audio analysis capabilities including:
//! - Time-frequency analysis (spectrograms, wavelets)  
//! - Perceptual analysis (loudness, critical bands)
//! - Statistical analysis (kurtosis, skewness, entropy)
//! - Feature extraction for machine learning

use crate::{AudioBuffer, Result, VocoderError};
use scirs2_core::ndarray::{Array1, Array2, s};
use scirs2_fft::{FftPlanner, RealFftPlanner};
use scirs2_core::Complex;
use std::f32::consts::PI;

pub mod spectrum;
pub mod spectrogram;
pub mod perceptual;
pub mod statistics;
pub mod features;

pub use spectrum::*;
pub use spectrogram::*;
pub use perceptual::*;
pub use statistics::*;
pub use features::*;

/// Comprehensive analysis configuration
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// FFT size for spectral analysis
    pub fft_size: usize,
    
    /// Hop length for time-frequency analysis
    pub hop_length: usize,
    
    /// Window function type
    pub window_type: WindowType,
    
    /// Number of mel filter banks
    pub n_mels: usize,
    
    /// Frequency range for analysis
    pub freq_range: (f32, f32),
    
    /// Enable perceptual weighting
    pub perceptual_weighting: bool,
    
    /// Statistical analysis depth
    pub statistics_depth: StatisticsDepth,
}

/// Window function types
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Kaiser(f32), // beta parameter
    Rectangular,
}

/// Statistical analysis depth levels
#[derive(Debug, Clone, Copy)]
pub enum StatisticsDepth {
    Basic,      // Mean, variance, min, max
    Extended,   // + skewness, kurtosis
    Full,       // + entropy, complexity measures
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_length: 512,
            window_type: WindowType::Hann,
            n_mels: 80,
            freq_range: (0.0, 8000.0),
            perceptual_weighting: true,
            statistics_depth: StatisticsDepth::Extended,
        }
    }
}

/// Comprehensive audio analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Spectral analysis
    pub spectrum: SpectrumAnalysis,
    
    /// Spectrogram analysis
    pub spectrogram: SpectrogramAnalysis,
    
    /// Perceptual analysis
    pub perceptual: PerceptualAnalysis,
    
    /// Statistical analysis
    pub statistics: StatisticsAnalysis,
    
    /// Extracted features
    pub features: FeatureSet,
}

/// Main audio analyzer with advanced capabilities
pub struct AdvancedAudioAnalyzer {
    config: AnalysisConfig,
    sample_rate: u32,
    fft_planner: RealFftPlanner<f32>,
}

impl AdvancedAudioAnalyzer {
    /// Create new advanced audio analyzer
    pub fn new(sample_rate: u32, config: AnalysisConfig) -> Self {
        Self {
            config,
            sample_rate,
            fft_planner: RealFftPlanner::<f32>::new(),
        }
    }
    
    /// Perform comprehensive audio analysis
    pub fn analyze(&mut self, audio: &AudioBuffer) -> Result<AnalysisResult> {
        // Convert to mono if needed
        let mono_samples = self.to_mono(audio);
        
        // Perform different types of analysis
        let spectrum = self.analyze_spectrum(&mono_samples)?;
        let spectrogram = self.analyze_spectrogram(&mono_samples)?;
        let perceptual = self.analyze_perceptual(&mono_samples)?;
        let statistics = self.analyze_statistics(&mono_samples)?;
        let features = self.extract_features(&mono_samples, &spectrum, &spectrogram)?;
        
        Ok(AnalysisResult {
            spectrum,
            spectrogram,
            perceptual,
            statistics,
            features,
        })
    }
    
    /// Convert audio to mono
    fn to_mono(&self, audio: &AudioBuffer) -> Array1<f32> {
        let samples = audio.samples();
        let channels = audio.channels() as usize;
        
        if channels == 1 {
            Array1::from_vec(samples.to_vec())
        } else {
            // Mix down to mono by averaging channels
            let mono_len = samples.len() / channels;
            let mut mono = Array1::zeros(mono_len);
            
            for i in 0..mono_len {
                let mut sum = 0.0;
                for ch in 0..channels {
                    sum += samples[i * channels + ch];
                }
                mono[i] = sum / channels as f32;
            }
            
            mono
        }
    }
    
    /// Generate window function
    fn generate_window(&self, size: usize) -> Vec<f32> {
        match self.config.window_type {
            WindowType::Hann => {
                (0..size)
                    .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
                    .collect()
            }
            WindowType::Hamming => {
                (0..size)
                    .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos())
                    .collect()
            }
            WindowType::Blackman => {
                (0..size)
                    .map(|i| {
                        let n = i as f32 / (size - 1) as f32;
                        0.42 - 0.5 * (2.0 * PI * n).cos() + 0.08 * (4.0 * PI * n).cos()
                    })
                    .collect()
            }
            WindowType::Kaiser(beta) => {
                let i0_beta = bessel_i0(beta);
                (0..size)
                    .map(|i| {
                        let n = 2.0 * i as f32 / (size - 1) as f32 - 1.0;
                        bessel_i0(beta * (1.0 - n * n).sqrt()) / i0_beta
                    })
                    .collect()
            }
            WindowType::Rectangular => vec![1.0; size],
        }
    }
    
    /// Perform comprehensive spectrum analysis
    fn analyze_spectrum(&mut self, samples: &Array1<f32>) -> Result<SpectrumAnalysis> {
        if samples.is_empty() {
            return Ok(SpectrumAnalysis::default());
        }
        
        // Compute FFT spectrum
        let window = self.generate_window(self.config.fft_size.min(samples.len()));
        let windowed_samples: Vec<f32> = samples.iter()
            .take(window.len())
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        
        // Pad to FFT size if necessary
        let mut padded = windowed_samples;
        padded.resize(self.config.fft_size, 0.0);
        
        // Compute FFT
        let mut spectrum = vec![Complex::new(0.0, 0.0); self.config.fft_size / 2 + 1];
        self.fft.process(&mut padded, &mut spectrum);
        
        // Convert to magnitude spectrum
        let magnitude_spectrum: Vec<f32> = spectrum.iter()
            .map(|c| c.norm())
            .collect();
        
        // Compute phase spectrum  
        let phase_spectrum: Vec<f32> = spectrum.iter()
            .map(|c| c.arg())
            .collect();
        
        // Calculate spectral features
        let total_energy: f32 = magnitude_spectrum.iter().map(|&m| m * m).sum();
        let peak_frequency = self.find_peak_frequency(&magnitude_spectrum);
        let bandwidth = self.calculate_bandwidth(&magnitude_spectrum);
        
        Ok(SpectrumAnalysis {
            magnitude_spectrum,
            phase_spectrum,
            total_energy,
            peak_frequency,
            bandwidth,
            ..Default::default()
        })
    }
    
    /// Perform comprehensive spectrogram analysis
    fn analyze_spectrogram(&mut self, samples: &Array1<f32>) -> Result<SpectrogramAnalysis> {
        if samples.len() < self.config.fft_size {
            return Ok(SpectrogramAnalysis::default());
        }
        
        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let window = self.generate_window(fft_size);
        
        // Calculate number of frames
        let num_frames = (samples.len() - fft_size) / hop_length + 1;
        let mut spectrogram = Array2::<f32>::zeros((fft_size / 2 + 1, num_frames));
        
        // Generate spectrogram
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;
            let end = start + fft_size;
            
            if end <= samples.len() {
                // Window the frame
                let mut windowed_frame: Vec<f32> = samples.slice(s![start..end])
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| s * w)
                    .collect();
                
                // Compute FFT
                let mut spectrum = vec![Complex::new(0.0, 0.0); fft_size / 2 + 1];
                if let Ok(()) = self.fft.process(&mut windowed_frame, &mut spectrum) {
                    // Store magnitude spectrum
                    for (freq_idx, complex_val) in spectrum.iter().enumerate() {
                        spectrogram[[freq_idx, frame_idx]] = complex_val.norm();
                    }
                }
            }
        }
        
        // Calculate temporal and spectral characteristics
        let temporal_centroid = self.calculate_temporal_centroid(&spectrogram);
        let spectral_centroid = self.calculate_spectral_centroid_over_time(&spectrogram);
        let spectral_bandwidth = self.calculate_spectral_bandwidth_over_time(&spectrogram);
        
        Ok(SpectrogramAnalysis {
            spectrogram,
            temporal_centroid,
            spectral_centroid,
            spectral_bandwidth,
            ..Default::default()
        })
    }
    
    /// Perform perceptual analysis using psychoacoustic models
    fn analyze_perceptual(&self, samples: &Array1<f32>) -> Result<PerceptualAnalysis> {
        if samples.is_empty() {
            return Ok(PerceptualAnalysis::default());
        }
        
        // Calculate RMS loudness
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let loudness = if rms > 1e-10 {
            20.0 * rms.log10() // Convert to dB
        } else {
            -100.0 // Very quiet signal
        };
        
        // Estimate perceived sharpness (high frequency content)
        let window = self.generate_window(self.config.fft_size.min(samples.len()));
        let mut windowed_samples: Vec<f32> = samples.iter()
            .take(window.len())
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        windowed_samples.resize(self.config.fft_size, 0.0);
        
        // Simple sharpness estimation based on high-frequency energy
        let mut high_freq_energy = 0.0;
        let mut total_energy = 0.0;
        
        for i in 1..windowed_samples.len()-1 {
            let high_freq = windowed_samples[i] - windowed_samples[i-1];
            high_freq_energy += high_freq * high_freq;
            total_energy += windowed_samples[i] * windowed_samples[i];
        }
        
        let sharpness = if total_energy > 1e-10 {
            (high_freq_energy / total_energy).sqrt()
        } else {
            0.0
        };
        
        // Estimate roughness based on amplitude modulation
        let mut roughness = 0.0;
        if samples.len() > 10 {
            let mut am_variations = 0.0;
            for i in 5..samples.len()-5 {
                let local_rms = (samples.slice(s![i-5..i+5]).iter().map(|&x| x * x).sum::<f32>() / 10.0).sqrt();
                if i > 5 {
                    let prev_rms = (samples.slice(s![i-10..i]).iter().map(|&x| x * x).sum::<f32>() / 10.0).sqrt();
                    am_variations += (local_rms - prev_rms).abs();
                }
            }
            roughness = am_variations / (samples.len() - 10) as f32;
        }
        
        Ok(PerceptualAnalysis {
            loudness,
            sharpness,
            roughness,
            ..Default::default()
        })
    }
    
    /// Perform comprehensive statistical analysis
    fn analyze_statistics(&self, samples: &Array1<f32>) -> Result<StatisticsAnalysis> {
        if samples.is_empty() {
            return Ok(StatisticsAnalysis::default());
        }
        
        let n = samples.len() as f32;
        
        // Basic statistics
        let mean = samples.iter().sum::<f32>() / n;
        let min_val = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Variance calculation
        let variance = samples.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / n;
        let std_dev = variance.sqrt();
        
        // Extended statistics (if enabled)
        let (skewness, kurtosis) = if matches!(self.config.statistics_depth, StatisticsDepth::Extended | StatisticsDepth::Full) {
            let m3 = samples.iter()
                .map(|&x| (x - mean).powi(3))
                .sum::<f32>() / n;
            let m4 = samples.iter()
                .map(|&x| (x - mean).powi(4))
                .sum::<f32>() / n;
            
            let skew = if std_dev > 1e-10 {
                m3 / std_dev.powi(3)
            } else {
                0.0
            };
            
            let kurt = if variance > 1e-10 {
                m4 / variance.powi(2) - 3.0 // Excess kurtosis
            } else {
                0.0
            };
            
            (skew, kurt)
        } else {
            (0.0, 0.0)
        };
        
        // Full statistics (if enabled)
        let entropy = if matches!(self.config.statistics_depth, StatisticsDepth::Full) {
            self.calculate_signal_entropy(samples)
        } else {
            0.0
        };
        
        Ok(StatisticsAnalysis {
            mean,
            variance,
            std_dev,
            min_val,
            max_val,
            skewness,
            kurtosis,
            entropy,
            ..Default::default()
        })
    }
    
    /// Extract comprehensive feature set from analysis results
    fn extract_features(
        &self,
        samples: &Array1<f32>,
        spectrum: &SpectrumAnalysis,
        spectrogram: &SpectrogramAnalysis,
    ) -> Result<FeatureSet> {
        if samples.is_empty() {
            return Ok(FeatureSet::default());
        }
        
        // Extract basic temporal features
        let zero_crossing_rate = self.calculate_zero_crossing_rate(samples);
        let energy = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        
        // Use spectrum analysis for spectral features
        let spectral_centroid = spectrum.peak_frequency;
        let spectral_bandwidth = spectrum.bandwidth;
        let spectral_rolloff = self.calculate_spectral_rolloff(&spectrum.magnitude_spectrum);
        
        // Use spectrogram for temporal-spectral features
        let spectral_flux = self.calculate_spectral_flux(spectrogram);
        
        // Create feature vector combining all extracted features
        let feature_vector = vec![
            zero_crossing_rate,
            energy.sqrt(), // RMS energy
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flux,
        ];
        
        Ok(FeatureSet {
            feature_vector,
            zero_crossing_rate,
            energy,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flux,
            ..Default::default()
        })
    }
    
    // Helper methods
    
    /// Find peak frequency in magnitude spectrum
    fn find_peak_frequency(&self, magnitude_spectrum: &[f32]) -> f32 {
        let peak_bin = magnitude_spectrum.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        // Convert bin to frequency
        let nyquist = self.sample_rate as f32 / 2.0;
        (peak_bin as f32 / magnitude_spectrum.len() as f32) * nyquist
    }
    
    /// Calculate spectrum bandwidth
    fn calculate_bandwidth(&self, magnitude_spectrum: &[f32]) -> f32 {
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_width = nyquist / magnitude_spectrum.len() as f32;
        
        // Calculate spectral centroid
        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;
        
        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let frequency = i as f32 * bin_width;
            weighted_sum += frequency * magnitude;
            total_magnitude += magnitude;
        }
        
        let centroid = if total_magnitude > 1e-10 {
            weighted_sum / total_magnitude
        } else {
            0.0
        };
        
        // Calculate bandwidth as weighted deviation from centroid
        let mut bandwidth = 0.0;
        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let frequency = i as f32 * bin_width;
            bandwidth += (frequency - centroid) * (frequency - centroid) * magnitude;
        }
        
        if total_magnitude > 1e-10 {
            (bandwidth / total_magnitude).sqrt()
        } else {
            0.0
        }
    }
    
    /// Calculate temporal centroid of spectrogram
    fn calculate_temporal_centroid(&self, spectrogram: &Array2<f32>) -> f32 {
        let (_freq_bins, time_frames) = spectrogram.dim();
        
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;
        
        for time_idx in 0..time_frames {
            let frame_energy: f32 = spectrogram.column(time_idx).iter().sum();
            weighted_sum += time_idx as f32 * frame_energy;
            total_energy += frame_energy;
        }
        
        if total_energy > 1e-10 {
            weighted_sum / total_energy
        } else {
            0.0
        }
    }
    
    /// Calculate spectral centroid over time
    fn calculate_spectral_centroid_over_time(&self, spectrogram: &Array2<f32>) -> Vec<f32> {
        let (freq_bins, time_frames) = spectrogram.dim();
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_width = nyquist / freq_bins as f32;
        
        let mut centroids = Vec::with_capacity(time_frames);
        
        for time_idx in 0..time_frames {
            let frame = spectrogram.column(time_idx);
            
            let mut weighted_sum = 0.0;
            let mut total_magnitude = 0.0;
            
            for (freq_idx, &magnitude) in frame.iter().enumerate() {
                let frequency = freq_idx as f32 * bin_width;
                weighted_sum += frequency * magnitude;
                total_magnitude += magnitude;
            }
            
            let centroid = if total_magnitude > 1e-10 {
                weighted_sum / total_magnitude
            } else {
                0.0
            };
            
            centroids.push(centroid);
        }
        
        centroids
    }
    
    /// Calculate spectral bandwidth over time
    fn calculate_spectral_bandwidth_over_time(&self, spectrogram: &Array2<f32>) -> Vec<f32> {
        let (freq_bins, time_frames) = spectrogram.dim();
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_width = nyquist / freq_bins as f32;
        
        let centroids = self.calculate_spectral_centroid_over_time(spectrogram);
        let mut bandwidths = Vec::with_capacity(time_frames);
        
        for time_idx in 0..time_frames {
            let frame = spectrogram.column(time_idx);
            let centroid = centroids[time_idx];
            
            let mut bandwidth = 0.0;
            let mut total_magnitude = 0.0;
            
            for (freq_idx, &magnitude) in frame.iter().enumerate() {
                let frequency = freq_idx as f32 * bin_width;
                bandwidth += (frequency - centroid) * (frequency - centroid) * magnitude;
                total_magnitude += magnitude;
            }
            
            let bw = if total_magnitude > 1e-10 {
                (bandwidth / total_magnitude).sqrt()
            } else {
                0.0
            };
            
            bandwidths.push(bw);
        }
        
        bandwidths
    }
    
    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, samples: &Array1<f32>) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i-1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        
        zero_crossings as f32 / (samples.len() - 1) as f32
    }
    
    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, magnitude_spectrum: &[f32]) -> f32 {
        let total_energy: f32 = magnitude_spectrum.iter().map(|&m| m * m).sum();
        let rolloff_threshold = total_energy * 0.85; // 85% energy threshold
        
        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= rolloff_threshold {
                let nyquist = self.sample_rate as f32 / 2.0;
                return (i as f32 / magnitude_spectrum.len() as f32) * nyquist;
            }
        }
        
        // If we reach here, return Nyquist frequency
        self.sample_rate as f32 / 2.0
    }
    
    /// Calculate spectral flux
    fn calculate_spectral_flux(&self, spectrogram: &SpectrogramAnalysis) -> f32 {
        let (freq_bins, time_frames) = spectrogram.spectrogram.dim();
        
        if time_frames < 2 {
            return 0.0;
        }
        
        let mut total_flux = 0.0;
        
        for time_idx in 1..time_frames {
            let current_frame = spectrogram.spectrogram.column(time_idx);
            let previous_frame = spectrogram.spectrogram.column(time_idx - 1);
            
            let mut frame_flux = 0.0;
            for freq_idx in 0..freq_bins {
                let diff = current_frame[freq_idx] - previous_frame[freq_idx];
                if diff > 0.0 {
                    frame_flux += diff;
                }
            }
            
            total_flux += frame_flux;
        }
        
        total_flux / (time_frames - 1) as f32
    }
    
    /// Calculate signal entropy
    fn calculate_signal_entropy(&self, samples: &Array1<f32>) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        
        // Quantize samples into bins for entropy calculation
        let num_bins = 256;
        let min_val = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if (max_val - min_val).abs() < 1e-10 {
            return 0.0; // Constant signal has zero entropy
        }
        
        let bin_width = (max_val - min_val) / num_bins as f32;
        let mut bins = vec![0; num_bins];
        
        // Quantize samples
        for &sample in samples.iter() {
            let bin_idx = ((sample - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += 1;
        }
        
        // Calculate entropy
        let total_samples = samples.len() as f32;
        let mut entropy = 0.0;
        
        for count in bins.iter() {
            if *count > 0 {
                let probability = *count as f32 / total_samples;
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
}

/// Zero-order modified Bessel function of the first kind
fn bessel_i0(x: f32) -> f32 {
    let x = x.abs();
    if x < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / x;
        (x.exp() / x.sqrt()) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analyzer_creation() {
        let config = AnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(22050, config);
        assert_eq!(analyzer.sample_rate, 22050);
    }
    
    #[test]
    fn test_window_generation() {
        let config = AnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(22050, config);
        
        let hann = analyzer.generate_window(64);
        assert_eq!(hann.len(), 64);
        assert!((hann[0] - 0.0).abs() < 1e-6);
        assert!((hann[32] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_to_mono_conversion() {
        let config = AnalysisConfig::default();
        let analyzer = AdvancedAudioAnalyzer::new(22050, config);
        
        // Stereo audio: [L1, R1, L2, R2]
        let stereo_samples = vec![1.0, 0.0, 0.5, 0.5];
        let stereo_audio = AudioBuffer::new(stereo_samples, 22050, 2);
        
        let mono = analyzer.to_mono(&stereo_audio);
        
        // Should be averaged: [(1.0+0.0)/2, (0.5+0.5)/2]
        let expected = vec![0.5, 0.5];
        assert_eq!(mono.len(), expected.len());
        
        for (actual, expected) in mono.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_bessel_function() {
        // Test some known values
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
        assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-5);
    }
}