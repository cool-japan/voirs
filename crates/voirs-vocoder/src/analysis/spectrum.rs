//! Advanced spectrum analysis tools

use crate::{Result, VocoderError};
use ndarray::Array1;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

/// Comprehensive spectrum analysis result
#[derive(Debug, Clone)]
pub struct SpectrumAnalysis {
    /// Frequency bins (Hz)
    pub frequencies: Vec<f32>,
    
    /// Magnitude spectrum (linear)
    pub magnitudes: Vec<f32>,
    
    /// Magnitude spectrum (dB)
    pub magnitudes_db: Vec<f32>,
    
    /// Phase spectrum (radians)
    pub phases: Vec<f32>,
    
    /// Power spectral density
    pub power_spectrum: Vec<f32>,
    
    /// Spectral features
    pub features: SpectralFeatures,
}

/// Spectral features
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid (Hz)
    pub centroid: f32,
    
    /// Spectral spread (Hz)
    pub spread: f32,
    
    /// Spectral skewness
    pub skewness: f32,
    
    /// Spectral kurtosis
    pub kurtosis: f32,
    
    /// Spectral rolloff (Hz)
    pub rolloff: f32,
    
    /// Spectral flux
    pub flux: f32,
    
    /// Spectral flatness
    pub flatness: f32,
    
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    
    /// Bandwidth
    pub bandwidth: f32,
    
    /// Peak frequency
    pub peak_frequency: f32,
}

impl Default for SpectrumAnalysis {
    fn default() -> Self {
        Self {
            frequencies: Vec::new(),
            magnitudes: Vec::new(),
            magnitudes_db: Vec::new(),
            phases: Vec::new(),
            power_spectrum: Vec::new(),
            features: SpectralFeatures::default(),
        }
    }
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            centroid: 0.0,
            spread: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            rolloff: 0.0,
            flux: 0.0,
            flatness: 0.0,
            zero_crossing_rate: 0.0,
            bandwidth: 0.0,
            peak_frequency: 0.0,
        }
    }
}

/// Advanced spectrum analyzer
pub struct SpectrumAnalyzer {
    sample_rate: u32,
    fft_planner: RealFftPlanner<f32>,
}

impl SpectrumAnalyzer {
    /// Create new spectrum analyzer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_planner: RealFftPlanner::new(),
        }
    }
    
    /// Analyze spectrum with comprehensive features
    pub fn analyze(&mut self, samples: &Array1<f32>, window: &[f32]) -> Result<SpectrumAnalysis> {
        if samples.is_empty() {
            return Ok(SpectrumAnalysis::default());
        }
        
        let fft_size = window.len();
        
        // Apply window and compute FFT
        let spectrum = self.compute_fft(samples, window)?;
        
        // Extract frequency bins
        let frequencies: Vec<f32> = (0..spectrum.len())
            .map(|i| i as f32 * self.sample_rate as f32 / fft_size as f32)
            .collect();
        
        // Compute magnitudes
        let magnitudes: Vec<f32> = spectrum.iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();
        
        // Compute magnitude in dB
        let magnitudes_db: Vec<f32> = magnitudes.iter()
            .map(|&m| if m > 1e-10 { 20.0 * m.log10() } else { -200.0 })
            .collect();
        
        // Compute phases
        let phases: Vec<f32> = spectrum.iter()
            .map(|c| c.im.atan2(c.re))
            .collect();
        
        // Compute power spectrum
        let power_spectrum: Vec<f32> = magnitudes.iter()
            .map(|&m| m * m)
            .collect();
        
        // Calculate spectral features
        let features = self.calculate_spectral_features(
            &frequencies,
            &magnitudes,
            &power_spectrum,
            samples
        );
        
        Ok(SpectrumAnalysis {
            frequencies,
            magnitudes,
            magnitudes_db,
            phases,
            power_spectrum,
            features,
        })
    }
    
    /// Compute FFT with windowing
    fn compute_fft(&mut self, samples: &Array1<f32>, window: &[f32]) -> Result<Vec<Complex<f32>>> {
        let fft_size = window.len();
        let mut fft = self.fft_planner.plan_fft_forward(fft_size);
        
        // Prepare input with windowing
        let mut input = vec![0.0; fft_size];
        let copy_len = fft_size.min(samples.len());
        for i in 0..copy_len {
            input[i] = samples[i] * window[i];
        }
        
        // Compute FFT
        let mut output = fft.make_output_vec();
        fft.process(&mut input, &mut output)
            .map_err(|_| VocoderError::ProcessingError("FFT computation failed".to_string()))?;
        
        Ok(output)
    }
    
    /// Calculate comprehensive spectral features
    fn calculate_spectral_features(
        &self,
        frequencies: &[f32],
        magnitudes: &[f32],
        power_spectrum: &[f32],
        samples: &Array1<f32>,
    ) -> SpectralFeatures {
        let centroid = self.calculate_spectral_centroid(frequencies, magnitudes);
        let spread = self.calculate_spectral_spread(frequencies, magnitudes, centroid);
        let skewness = self.calculate_spectral_skewness(frequencies, magnitudes, centroid, spread);
        let kurtosis = self.calculate_spectral_kurtosis(frequencies, magnitudes, centroid, spread);
        let rolloff = self.calculate_spectral_rolloff(frequencies, magnitudes, 0.85);
        let flux = 0.0; // Requires previous frame for comparison
        let flatness = self.calculate_spectral_flatness(magnitudes);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(samples);
        let bandwidth = self.calculate_bandwidth(frequencies, magnitudes);
        let peak_frequency = self.find_peak_frequency(frequencies, magnitudes);
        
        SpectralFeatures {
            centroid,
            spread,
            skewness,
            kurtosis,
            rolloff,
            flux,
            flatness,
            zero_crossing_rate,
            bandwidth,
            peak_frequency,
        }
    }
    
    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            weighted_sum += freq * mag;
            magnitude_sum += mag;
        }
        
        if magnitude_sum > 1e-10 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
    
    /// Calculate spectral spread (standard deviation around centroid)
    fn calculate_spectral_spread(&self, frequencies: &[f32], magnitudes: &[f32], centroid: f32) -> f32 {
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
    
    /// Calculate spectral skewness
    fn calculate_spectral_skewness(&self, frequencies: &[f32], magnitudes: &[f32], centroid: f32, spread: f32) -> f32 {
        if spread <= 1e-10 {
            return 0.0;
        }
        
        let mut weighted_skewness = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            let normalized_diff = (freq - centroid) / spread;
            weighted_skewness += normalized_diff.powi(3) * mag;
            magnitude_sum += mag;
        }
        
        if magnitude_sum > 1e-10 {
            weighted_skewness / magnitude_sum
        } else {
            0.0
        }
    }
    
    /// Calculate spectral kurtosis
    fn calculate_spectral_kurtosis(&self, frequencies: &[f32], magnitudes: &[f32], centroid: f32, spread: f32) -> f32 {
        if spread <= 1e-10 {
            return 0.0;
        }
        
        let mut weighted_kurtosis = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (freq, mag) in frequencies.iter().zip(magnitudes.iter()) {
            let normalized_diff = (freq - centroid) / spread;
            weighted_kurtosis += normalized_diff.powi(4) * mag;
            magnitude_sum += mag;
        }
        
        if magnitude_sum > 1e-10 {
            weighted_kurtosis / magnitude_sum - 3.0 // Subtract 3 for excess kurtosis
        } else {
            0.0
        }
    }
    
    /// Calculate spectral rolloff (frequency below which given percentage of energy is contained)
    fn calculate_spectral_rolloff(&self, frequencies: &[f32], magnitudes: &[f32], threshold: f32) -> f32 {
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
    
    /// Calculate spectral flatness (measure of noise-like vs tonal quality)
    fn calculate_spectral_flatness(&self, magnitudes: &[f32]) -> f32 {
        if magnitudes.len() <= 1 {
            return 0.0;
        }
        
        // Skip DC component
        let relevant_mags = &magnitudes[1..];
        
        // Geometric mean
        let log_sum: f32 = relevant_mags.iter()
            .map(|&m| if m > 1e-10 { m.ln() } else { -23.0 }) // -23 ≈ ln(1e-10)
            .sum();
        let geometric_mean = (log_sum / relevant_mags.len() as f32).exp();
        
        // Arithmetic mean
        let arithmetic_mean: f32 = relevant_mags.iter().sum::<f32>() / relevant_mags.len() as f32;
        
        if arithmetic_mean > 1e-10 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
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
        
        zero_crossings as f32 / samples.len() as f32
    }
    
    /// Calculate bandwidth (weighted standard deviation of frequencies)
    fn calculate_bandwidth(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
        let centroid = self.calculate_spectral_centroid(frequencies, magnitudes);
        self.calculate_spectral_spread(frequencies, magnitudes, centroid)
    }
    
    /// Find peak frequency (frequency with maximum magnitude)
    fn find_peak_frequency(&self, frequencies: &[f32], magnitudes: &[f32]) -> f32 {
        if let Some((idx, _)) = magnitudes.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
            frequencies.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_spectrum_analyzer_creation() {
        let analyzer = SpectrumAnalyzer::new(22050);
        assert_eq!(analyzer.sample_rate, 22050);
    }
    
    #[test]
    fn test_spectral_centroid() {
        let analyzer = SpectrumAnalyzer::new(22050);
        
        // Create simple test spectrum
        let frequencies = vec![100.0, 200.0, 300.0, 400.0];
        let magnitudes = vec![0.1, 1.0, 0.1, 0.1]; // Peak at 200 Hz
        
        let centroid = analyzer.calculate_spectral_centroid(&frequencies, &magnitudes);
        
        // Should be close to 200 Hz since that's where the peak is
        assert!((centroid - 200.0).abs() < 50.0);
    }
    
    #[test]
    fn test_spectral_rolloff() {
        let analyzer = SpectrumAnalyzer::new(22050);
        
        let frequencies = vec![100.0, 200.0, 300.0, 400.0];
        let magnitudes = vec![1.0, 1.0, 0.1, 0.1]; // Most energy in lower frequencies
        
        let rolloff = analyzer.calculate_spectral_rolloff(&frequencies, &magnitudes, 0.85);
        
        // Should be around 200-300 Hz
        assert!(rolloff >= 200.0 && rolloff <= 400.0);
    }
    
    #[test]
    fn test_spectral_flatness() {
        let analyzer = SpectrumAnalyzer::new(22050);
        
        // Flat spectrum (white noise-like)
        let flat_mags = vec![1.0; 10];
        let flatness_flat = analyzer.calculate_spectral_flatness(&flat_mags);
        
        // Peaked spectrum (tonal)
        let peaked_mags = vec![0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let flatness_peaked = analyzer.calculate_spectral_flatness(&peaked_mags);
        
        // Flat spectrum should have higher flatness
        assert!(flatness_flat > flatness_peaked);
    }
    
    #[test]
    fn test_zero_crossing_rate() {
        let analyzer = SpectrumAnalyzer::new(22050);
        
        // Alternating signal
        let samples = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0]);
        let zcr = analyzer.calculate_zero_crossing_rate(&samples);
        
        // Should have high ZCR
        assert!(zcr > 0.5);
    }
    
    #[test]
    fn test_peak_frequency() {
        let analyzer = SpectrumAnalyzer::new(22050);
        
        let frequencies = vec![100.0, 200.0, 300.0, 400.0];
        let magnitudes = vec![0.1, 0.2, 1.0, 0.1]; // Peak at 300 Hz
        
        let peak_freq = analyzer.find_peak_frequency(&frequencies, &magnitudes);
        
        assert!((peak_freq - 300.0).abs() < 1e-6);
    }
}