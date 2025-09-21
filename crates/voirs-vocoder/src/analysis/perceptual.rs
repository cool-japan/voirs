//! Perceptual audio analysis tools
//!
//! Implements psychoacoustic models and perceptual metrics including:
//! - LUFS (Loudness Units relative to Full Scale)
//! - Bark scale analysis
//! - Masking threshold computation
//! - Critical band analysis

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2, s};
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

/// Comprehensive perceptual analysis result
#[derive(Debug, Clone)]
pub struct PerceptualAnalysis {
    /// LUFS loudness measurement
    pub loudness_lufs: f32,
    
    /// Integrated loudness over time
    pub integrated_loudness: f32,
    
    /// Short-term loudness (3 seconds)
    pub short_term_loudness: Vec<f32>,
    
    /// Momentary loudness (400ms)
    pub momentary_loudness: Vec<f32>,
    
    /// Loudness range (LRA)
    pub loudness_range: f32,
    
    /// Bark scale representation
    pub bark_spectrum: Vec<f32>,
    
    /// Critical band energies
    pub critical_band_energies: Vec<f32>,
    
    /// Masking threshold
    pub masking_threshold: Vec<f32>,
    
    /// Perceptual features
    pub perceptual_features: PerceptualFeatures,
}

/// Perceptual audio features
#[derive(Debug, Clone)]
pub struct PerceptualFeatures {
    /// Roughness (sensory dissonance)
    pub roughness: f32,
    
    /// Sharpness (high-frequency emphasis)
    pub sharpness: f32,
    
    /// Fluctuation strength (amplitude modulation perception)
    pub fluctuation_strength: f32,
    
    /// Tonality (tonal vs noise-like character)
    pub tonality: f32,
    
    /// Brightness (spectral centroid in perceptual scale)
    pub brightness: f32,
    
    /// Warmth (low-frequency emphasis)
    pub warmth: f32,
    
    /// Fullness (mid-frequency content)
    pub fullness: f32,
    
    /// Clarity (high-frequency detail)
    pub clarity: f32,
}

impl Default for PerceptualAnalysis {
    fn default() -> Self {
        Self {
            loudness_lufs: -70.0, // Digital silence
            integrated_loudness: -70.0,
            short_term_loudness: Vec::new(),
            momentary_loudness: Vec::new(),
            loudness_range: 0.0,
            bark_spectrum: Vec::new(),
            critical_band_energies: Vec::new(),
            masking_threshold: Vec::new(),
            perceptual_features: PerceptualFeatures::default(),
        }
    }
}

impl Default for PerceptualFeatures {
    fn default() -> Self {
        Self {
            roughness: 0.0,
            sharpness: 0.0,
            fluctuation_strength: 0.0,
            tonality: 0.0,
            brightness: 0.0,
            warmth: 0.0,
            fullness: 0.0,
            clarity: 0.0,
        }
    }
}

/// Perceptual audio analyzer
pub struct PerceptualAnalyzer {
    sample_rate: u32,
    fft_planner: RealFftPlanner<f32>,
    bark_filterbank: BarkFilterbank,
    critical_bands: CriticalBands,
}

impl PerceptualAnalyzer {
    /// Create new perceptual analyzer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_planner: RealFftPlanner::new(),
            bark_filterbank: BarkFilterbank::new(sample_rate),
            critical_bands: CriticalBands::new(sample_rate),
        }
    }
    
    /// Perform comprehensive perceptual analysis
    pub fn analyze(&mut self, samples: &Array1<f32>) -> Result<PerceptualAnalysis> {
        if samples.is_empty() {
            return Ok(PerceptualAnalysis::default());
        }
        
        // LUFS loudness measurement
        let loudness_lufs = self.calculate_lufs_loudness(samples)?;
        
        // Time-varying loudness measurements
        let (short_term_loudness, momentary_loudness) = self.calculate_time_varying_loudness(samples)?;
        
        // Integrated loudness and range
        let integrated_loudness = self.calculate_integrated_loudness(&short_term_loudness);
        let loudness_range = self.calculate_loudness_range(&short_term_loudness);
        
        // Bark scale analysis
        let bark_spectrum = self.analyze_bark_spectrum(samples)?;
        
        // Critical band analysis
        let critical_band_energies = self.analyze_critical_bands(samples)?;
        
        // Masking threshold computation
        let masking_threshold = self.compute_masking_threshold(&critical_band_energies)?;
        
        // Perceptual features
        let perceptual_features = self.extract_perceptual_features(samples, &bark_spectrum)?;
        
        Ok(PerceptualAnalysis {
            loudness_lufs,
            integrated_loudness,
            short_term_loudness,
            momentary_loudness,
            loudness_range,
            bark_spectrum,
            critical_band_energies,
            masking_threshold,
            perceptual_features,
        })
    }
    
    /// Calculate LUFS loudness according to ITU-R BS.1770-4
    fn calculate_lufs_loudness(&mut self, samples: &Array1<f32>) -> Result<f32> {
        // Apply K-weighting filter (simplified implementation)
        let k_weighted = self.apply_k_weighting(samples)?;
        
        // Calculate mean square value
        let mean_square: f32 = k_weighted.iter().map(|&x| x * x).sum::<f32>() / k_weighted.len() as f32;
        
        // Convert to LUFS
        if mean_square > 1e-10 {
            Ok(-0.691 + 10.0 * mean_square.log10())
        } else {
            Ok(-70.0) // Digital silence
        }
    }
    
    /// Apply K-weighting filter (simplified)
    fn apply_k_weighting(&mut self, samples: &Array1<f32>) -> Result<Array1<f32>> {
        // Simplified K-weighting: high-pass + high-shelf filter
        // This is a basic approximation - full implementation would use IIR filters
        
        let mut filtered = samples.clone();
        
        // Simple high-pass filter at ~38 Hz
        let rc = 1.0 / (2.0 * PI * 38.0);
        let dt = 1.0 / self.sample_rate as f32;
        let alpha = rc / (rc + dt);
        
        let mut prev_input = 0.0;
        let mut prev_output = 0.0;
        
        for i in 0..filtered.len() {
            let output = alpha * (prev_output + filtered[i] - prev_input);
            prev_input = filtered[i];
            prev_output = output;
            filtered[i] = output;
        }
        
        // Apply high-shelf filter at ~1.5 kHz (simplified)
        // In practice, this would be a proper biquad filter
        let shelf_gain = 1.53; // ~4 dB boost
        for sample in filtered.iter_mut() {
            *sample *= shelf_gain;
        }
        
        Ok(filtered)
    }
    
    /// Calculate time-varying loudness
    fn calculate_time_varying_loudness(&mut self, samples: &Array1<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
        let short_term_window = (3.0 * self.sample_rate as f32) as usize; // 3 seconds
        let momentary_window = (0.4 * self.sample_rate as f32) as usize; // 400ms
        let hop_length = (0.1 * self.sample_rate as f32) as usize; // 100ms hop
        
        let mut short_term_loudness = Vec::new();
        let mut momentary_loudness = Vec::new();
        
        let mut pos = 0;
        while pos + momentary_window <= samples.len() {
            // Momentary loudness
            let momentary_chunk = samples.slice(s![pos..pos + momentary_window]);
            let momentary_lufs = self.calculate_lufs_loudness(&momentary_chunk.to_owned())?;
            momentary_loudness.push(momentary_lufs);
            
            // Short-term loudness (if we have enough samples)
            if pos + short_term_window <= samples.len() {
                let short_term_chunk = samples.slice(s![pos..pos + short_term_window]);
                let short_term_lufs = self.calculate_lufs_loudness(&short_term_chunk.to_owned())?;
                short_term_loudness.push(short_term_lufs);
            }
            
            pos += hop_length;
        }
        
        Ok((short_term_loudness, momentary_loudness))
    }
    
    /// Calculate integrated loudness
    fn calculate_integrated_loudness(&self, short_term_loudness: &[f32]) -> f32 {
        if short_term_loudness.is_empty() {
            return -70.0;
        }
        
        // Remove values below absolute threshold (-70 LUFS)
        let above_threshold: Vec<f32> = short_term_loudness.iter()
            .copied()
            .filter(|&x| x > -70.0)
            .collect();
        
        if above_threshold.is_empty() {
            return -70.0;
        }
        
        // Calculate relative threshold (10 LU below mean)
        let mean_loudness: f32 = above_threshold.iter().sum::<f32>() / above_threshold.len() as f32;
        let relative_threshold = mean_loudness - 10.0;
        
        // Calculate integrated loudness from values above relative threshold
        let above_relative: Vec<f32> = above_threshold.iter()
            .copied()
            .filter(|&x| x > relative_threshold)
            .collect();
        
        if above_relative.is_empty() {
            mean_loudness
        } else {
            above_relative.iter().sum::<f32>() / above_relative.len() as f32
        }
    }
    
    /// Calculate loudness range (LRA)
    fn calculate_loudness_range(&self, short_term_loudness: &[f32]) -> f32 {
        if short_term_loudness.len() < 2 {
            return 0.0;
        }
        
        // Filter values above absolute threshold
        let mut valid_values: Vec<f32> = short_term_loudness.iter()
            .copied()
            .filter(|&x| x > -70.0)
            .collect();
        
        if valid_values.len() < 2 {
            return 0.0;
        }
        
        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate 10th and 95th percentiles
        let p10_idx = (valid_values.len() as f32 * 0.10) as usize;
        let p95_idx = (valid_values.len() as f32 * 0.95) as usize;
        
        let p10 = valid_values[p10_idx.min(valid_values.len() - 1)];
        let p95 = valid_values[p95_idx.min(valid_values.len() - 1)];
        
        p95 - p10
    }
    
    /// Analyze spectrum in Bark scale
    fn analyze_bark_spectrum(&mut self, samples: &Array1<f32>) -> Result<Vec<f32>> {
        // Compute magnitude spectrum
        let fft_size = 2048;
        let mut fft = self.fft_planner.plan_fft_forward(fft_size);
        let mut input = vec![0.0; fft_size];
        let mut output = fft.make_output_vec();
        
        // Copy samples (zero-pad if necessary)
        let copy_len = fft_size.min(samples.len());
        for i in 0..copy_len {
            input[i] = samples[i];
        }
        
        // Apply window
        for i in 0..fft_size {
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos());
            input[i] *= window_val;
        }
        
        fft.process(&mut input, &mut output)
            .map_err(|_| VocoderError::ProcessingError("FFT computation failed".to_string()))?;
        
        // Convert to magnitude spectrum
        let magnitude_spectrum: Vec<f32> = output.iter()
            .map(|c| c.norm())
            .collect();
        
        // Apply Bark filterbank
        self.bark_filterbank.apply(&magnitude_spectrum)
    }
    
    /// Analyze critical bands
    fn analyze_critical_bands(&mut self, samples: &Array1<f32>) -> Result<Vec<f32>> {
        // Similar to Bark analysis but using critical band filterbank
        let fft_size = 2048;
        let mut fft = self.fft_planner.plan_fft_forward(fft_size);
        let mut input = vec![0.0; fft_size];
        let mut output = fft.make_output_vec();
        
        let copy_len = fft_size.min(samples.len());
        for i in 0..copy_len {
            input[i] = samples[i];
        }
        
        // Apply window
        for i in 0..fft_size {
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos());
            input[i] *= window_val;
        }
        
        fft.process(&mut input, &mut output)
            .map_err(|_| VocoderError::ProcessingError("FFT computation failed".to_string()))?;
        
        let power_spectrum: Vec<f32> = output.iter()
            .map(|c| c.norm_sqr())
            .collect();
        
        self.critical_bands.analyze(&power_spectrum)
    }
    
    /// Compute masking threshold
    fn compute_masking_threshold(&self, critical_band_energies: &[f32]) -> Result<Vec<f32>> {
        let mut masking_threshold = vec![0.0; critical_band_energies.len()];
        
        // Simplified masking model
        for (i, &energy) in critical_band_energies.iter().enumerate() {
            if energy > 1e-10 {
                // Spreading function (simplified)
                let energy_db = 10.0 * energy.log10();
                
                for (j, threshold) in masking_threshold.iter_mut().enumerate() {
                    let distance = (i as f32 - j as f32).abs();
                    
                    // Simple spreading function
                    let spread_factor = if distance <= 1.0 {
                        1.0
                    } else {
                        (-0.6 * distance).exp()
                    };
                    
                    // Masking contribution
                    let masking_contribution = energy_db - 14.5 - distance * 3.0;
                    let masked_threshold = 10.0_f32.powf(masking_contribution / 10.0) * spread_factor;
                    
                    *threshold += masked_threshold;
                }
            }
        }
        
        // Add absolute hearing threshold
        for (i, threshold) in masking_threshold.iter_mut().enumerate() {
            let abs_threshold = self.absolute_hearing_threshold(i);
            *threshold = threshold.max(abs_threshold);
        }
        
        Ok(masking_threshold)
    }
    
    /// Get absolute hearing threshold for critical band
    fn absolute_hearing_threshold(&self, band_index: usize) -> f32 {
        // Simplified absolute threshold curve
        let freq = self.critical_bands.center_frequency(band_index);
        
        // Approximation of absolute threshold in quiet (dB SPL)
        let threshold_db = if freq < 1000.0 {
            3.64 * (freq / 1000.0).powf(0.8) - 6.5 * (-0.6 * (freq / 1000.0 - 3.3).powi(2)).exp() + 1e-3 * (freq / 1000.0).powi(4)
        } else {
            3.64 * (freq / 1000.0).powf(0.8) - 6.5 * (-0.6 * (freq / 1000.0 - 3.3).powi(2)).exp() + 1e-3 * (freq / 1000.0).powi(4)
        };
        
        // Convert to linear scale (simplified)
        10.0_f32.powf(threshold_db / 20.0) * 1e-6
    }
    
    /// Extract perceptual features
    fn extract_perceptual_features(&self, samples: &Array1<f32>, bark_spectrum: &[f32]) -> Result<PerceptualFeatures> {
        let roughness = self.calculate_roughness(bark_spectrum);
        let sharpness = self.calculate_sharpness(bark_spectrum);
        let fluctuation_strength = self.calculate_fluctuation_strength(samples)?;
        let tonality = self.calculate_tonality(bark_spectrum);
        let brightness = self.calculate_brightness(bark_spectrum);
        let warmth = self.calculate_warmth(bark_spectrum);
        let fullness = self.calculate_fullness(bark_spectrum);
        let clarity = self.calculate_clarity(bark_spectrum);
        
        Ok(PerceptualFeatures {
            roughness,
            sharpness,
            fluctuation_strength,
            tonality,
            brightness,
            warmth,
            fullness,
            clarity,
        })
    }
    
    /// Calculate roughness (sensory dissonance)
    fn calculate_roughness(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.len() < 2 {
            return 0.0;
        }
        
        let mut roughness = 0.0;
        
        // Adjacent critical band interactions
        for i in 0..bark_spectrum.len()-1 {
            let level1 = bark_spectrum[i];
            let level2 = bark_spectrum[i + 1];
            
            if level1 > 1e-10 && level2 > 1e-10 {
                // Simplified roughness calculation
                let interaction = (level1 * level2).sqrt();
                roughness += interaction * 0.25; // Scaling factor
            }
        }
        
        roughness
    }
    
    /// Calculate sharpness (high-frequency emphasis)
    fn calculate_sharpness(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.is_empty() {
            return 0.0;
        }
        
        let mut weighted_sum = 0.0;
        let mut total_loudness = 0.0;
        
        for (i, &level) in bark_spectrum.iter().enumerate() {
            // Sharpness weighting function (increases with frequency)
            let weight = if i < bark_spectrum.len() / 2 {
                1.0
            } else {
                1.0 + 3.0 * ((i as f32 / bark_spectrum.len() as f32) - 0.5)
            };
            
            weighted_sum += level * weight;
            total_loudness += level;
        }
        
        if total_loudness > 1e-10 {
            weighted_sum / total_loudness
        } else {
            0.0
        }
    }
    
    /// Calculate fluctuation strength (amplitude modulation perception)
    fn calculate_fluctuation_strength(&self, samples: &Array1<f32>) -> Result<f32> {
        if samples.len() < 1024 {
            return Ok(0.0);
        }
        
        // Compute envelope using moving average
        let window_size = self.sample_rate as usize / 100; // 10ms window
        let mut envelope = Vec::new();
        
        for i in 0..samples.len().saturating_sub(window_size) {
            let chunk = &samples.as_slice().unwrap()[i..i + window_size];
            let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            envelope.push(rms);
        }
        
        if envelope.len() < 10 {
            return Ok(0.0);
        }
        
        // Analyze modulation frequency content (simplified)
        let mut modulation_energy = 0.0;
        for i in 1..envelope.len() {
            let diff = envelope[i] - envelope[i-1];
            modulation_energy += diff * diff;
        }
        
        Ok(modulation_energy / envelope.len() as f32)
    }
    
    /// Calculate tonality
    fn calculate_tonality(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.len() < 3 {
            return 0.0;
        }
        
        // Find peaks in bark spectrum
        let mut peak_energy = 0.0;
        let mut total_energy = 0.0;
        
        for i in 1..bark_spectrum.len()-1 {
            total_energy += bark_spectrum[i];
            
            // Check if this is a peak
            if bark_spectrum[i] > bark_spectrum[i-1] && bark_spectrum[i] > bark_spectrum[i+1] {
                peak_energy += bark_spectrum[i];
            }
        }
        
        if total_energy > 1e-10 {
            peak_energy / total_energy
        } else {
            0.0
        }
    }
    
    /// Calculate brightness
    fn calculate_brightness(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.is_empty() {
            return 0.0;
        }
        
        let cutoff = bark_spectrum.len() * 2 / 3; // Upper third of spectrum
        let high_freq_energy: f32 = bark_spectrum[cutoff..].iter().sum();
        let total_energy: f32 = bark_spectrum.iter().sum();
        
        if total_energy > 1e-10 {
            high_freq_energy / total_energy
        } else {
            0.0
        }
    }
    
    /// Calculate warmth
    fn calculate_warmth(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.is_empty() {
            return 0.0;
        }
        
        let cutoff = bark_spectrum.len() / 3; // Lower third of spectrum
        let low_freq_energy: f32 = bark_spectrum[..cutoff].iter().sum();
        let total_energy: f32 = bark_spectrum.iter().sum();
        
        if total_energy > 1e-10 {
            low_freq_energy / total_energy
        } else {
            0.0
        }
    }
    
    /// Calculate fullness
    fn calculate_fullness(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.is_empty() {
            return 0.0;
        }
        
        // Middle frequency energy
        let start = bark_spectrum.len() / 4;
        let end = bark_spectrum.len() * 3 / 4;
        let mid_freq_energy: f32 = bark_spectrum[start..end].iter().sum();
        let total_energy: f32 = bark_spectrum.iter().sum();
        
        if total_energy > 1e-10 {
            mid_freq_energy / total_energy
        } else {
            0.0
        }
    }
    
    /// Calculate clarity
    fn calculate_clarity(&self, bark_spectrum: &[f32]) -> f32 {
        if bark_spectrum.len() < 4 {
            return 0.0;
        }
        
        // High-frequency detail (upper quarter)
        let cutoff = bark_spectrum.len() * 3 / 4;
        let high_detail_energy: f32 = bark_spectrum[cutoff..].iter().sum();
        let total_energy: f32 = bark_spectrum.iter().sum();
        
        if total_energy > 1e-10 {
            high_detail_energy / total_energy
        } else {
            0.0
        }
    }
}

/// Bark scale filterbank
struct BarkFilterbank {
    sample_rate: u32,
    num_bands: usize,
}

impl BarkFilterbank {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            num_bands: 24, // Standard 24 Bark bands
        }
    }
    
    fn apply(&self, magnitude_spectrum: &[f32]) -> Result<Vec<f32>> {
        let mut bark_spectrum = vec![0.0; self.num_bands];
        
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_freq_step = nyquist / magnitude_spectrum.len() as f32;
        
        for (bin_idx, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let freq = bin_idx as f32 * bin_freq_step;
            let bark_value = self.hz_to_bark(freq);
            
            // Find corresponding Bark band
            let band_idx = (bark_value.floor() as usize).min(self.num_bands - 1);
            bark_spectrum[band_idx] += magnitude;
        }
        
        Ok(bark_spectrum)
    }
    
    fn hz_to_bark(&self, freq_hz: f32) -> f32 {
        // Zwicker & Terhardt formula
        13.0 * (0.76 * freq_hz / 1000.0).atan() + 3.5 * ((freq_hz / 7500.0).powi(2)).atan()
    }
}

/// Critical bands analyzer
struct CriticalBands {
    sample_rate: u32,
    num_bands: usize,
    center_frequencies: Vec<f32>,
}

impl CriticalBands {
    fn new(sample_rate: u32) -> Self {
        let center_frequencies = vec![
            50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0,
            1370.0, 1600.0, 1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0,
            5800.0, 7000.0, 8500.0, 10500.0, 13500.0
        ];
        
        Self {
            sample_rate,
            num_bands: center_frequencies.len(),
            center_frequencies,
        }
    }
    
    fn analyze(&self, power_spectrum: &[f32]) -> Result<Vec<f32>> {
        let mut band_energies = vec![0.0; self.num_bands];
        
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_freq_step = nyquist / power_spectrum.len() as f32;
        
        for (bin_idx, &power) in power_spectrum.iter().enumerate() {
            let freq = bin_idx as f32 * bin_freq_step;
            
            // Find closest critical band
            let band_idx = self.find_closest_band(freq);
            band_energies[band_idx] += power;
        }
        
        Ok(band_energies)
    }
    
    fn find_closest_band(&self, freq: f32) -> usize {
        self.center_frequencies
            .iter()
            .enumerate()
            .min_by_key(|(_, &center)| ((center - freq).abs() * 1000.0) as i32)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    fn center_frequency(&self, band_index: usize) -> f32 {
        self.center_frequencies.get(band_index).copied().unwrap_or(1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_perceptual_analyzer_creation() {
        let analyzer = PerceptualAnalyzer::new(44100);
        assert_eq!(analyzer.sample_rate, 44100);
    }
    
    #[test]
    fn test_bark_conversion() {
        let filterbank = BarkFilterbank::new(44100);
        
        // Test known conversions
        let bark_1000 = filterbank.hz_to_bark(1000.0);
        assert!((bark_1000 - 8.5).abs() < 1.0); // Approximately 8.5 Bark
    }
    
    #[test]
    fn test_lufs_calculation() {
        let mut analyzer = PerceptualAnalyzer::new(44100);
        
        // Test digital silence
        let silence = Array1::zeros(44100);
        let lufs = analyzer.calculate_lufs_loudness(&silence).unwrap();
        assert!(lufs < -60.0); // Should be very low
        
        // Test non-zero signal
        let sine_wave: Array1<f32> = Array1::from_vec(
            (0..44100).map(|i| 0.1 * (2.0 * PI * 1000.0 * i as f32 / 44100.0).sin()).collect()
        );
        let lufs_sine = analyzer.calculate_lufs_loudness(&sine_wave).unwrap();
        assert!(lufs_sine > lufs); // Should be louder than silence
    }
    
    #[test]
    fn test_critical_bands() {
        let critical_bands = CriticalBands::new(44100);
        assert_eq!(critical_bands.num_bands, 24);
        
        // Test frequency mapping
        let band_1000 = critical_bands.find_closest_band(1000.0);
        assert!(band_1000 < critical_bands.num_bands);
    }
    
    #[test]
    fn test_perceptual_features() {
        let analyzer = PerceptualAnalyzer::new(44100);
        
        // Test with flat spectrum
        let flat_spectrum = vec![1.0; 24];
        let brightness = analyzer.calculate_brightness(&flat_spectrum);
        let warmth = analyzer.calculate_warmth(&flat_spectrum);
        
        // Flat spectrum should have balanced brightness and warmth
        assert!((brightness - 0.33).abs() < 0.1); // Roughly 1/3 for upper third
        assert!((warmth - 0.33).abs() < 0.1); // Roughly 1/3 for lower third
    }
    
    #[test]
    fn test_loudness_range_calculation() {
        let analyzer = PerceptualAnalyzer::new(44100);
        
        // Test with varying loudness values
        let loudness_values = vec![-30.0, -25.0, -20.0, -15.0, -10.0];
        let lra = analyzer.calculate_loudness_range(&loudness_values);
        
        // Should be positive and reasonable
        assert!(lra > 0.0 && lra < 30.0);
    }
}