//! Vibrato processor for singing voice vocoder.

use crate::models::singing::config::VibratoConfig;
use anyhow::Result;
use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;

/// Processor for vibrato detection and enhancement
pub struct VibratoProcessor {
    /// Configuration
    config: VibratoConfig,
    /// Vibrato history buffer
    vibrato_history: VecDeque<VibratoFrame>,
    /// FFT planner for analysis
    fft_planner: FftPlanner<f32>,
    /// Window size for analysis
    window_size: usize,
    /// Hop size for analysis
    hop_size: usize,
    /// Sample rate
    sample_rate: u32,
}

/// Frame containing vibrato information
#[derive(Debug, Clone)]
struct VibratoFrame {
    /// Vibrato rate (Hz)
    rate: f32,
    /// Vibrato depth (cents)
    depth: f32,
    /// Vibrato strength (0.0-1.0)
    strength: f32,
    /// Frame timestamp
    timestamp: f32,
}

impl VibratoProcessor {
    /// Create new vibrato processor
    pub fn new(config: &VibratoConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            vibrato_history: VecDeque::with_capacity(50),
            fft_planner: FftPlanner::new(),
            window_size: 2048,
            hop_size: 512,
            sample_rate: 22050,
        })
    }

    /// Process mel spectrogram for vibrato enhancement
    pub fn process(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.config.enable_enhancement {
            return Ok(mel_spectrogram.clone());
        }

        let mut processed = mel_spectrogram.clone();
        let frames = mel_spectrogram.shape()[1];

        // Process each frame
        for frame_idx in 0..frames {
            let frame = mel_spectrogram.column(frame_idx);
            let vibrato_info = self.analyze_vibrato(&frame, frame_idx)?;

            // Apply vibrato enhancement
            self.apply_vibrato_enhancement(&mut processed, frame_idx, &vibrato_info)?;

            // Store vibrato information
            self.vibrato_history.push_back(vibrato_info);
            if self.vibrato_history.len() > 50 {
                self.vibrato_history.pop_front();
            }
        }

        Ok(processed)
    }

    /// Analyze vibrato characteristics in a frame
    fn analyze_vibrato(
        &mut self,
        frame: &ndarray::ArrayView1<f32>,
        frame_idx: usize,
    ) -> Result<VibratoFrame> {
        // Convert frame to frequency domain
        let mut spectrum = self.mel_to_spectrum(frame)?;

        // Apply windowing
        self.apply_hann_window(&mut spectrum);

        // Perform FFT
        let mut fft_input: Vec<Complex<f32>> =
            spectrum.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let fft = self.fft_planner.plan_fft_forward(fft_input.len());
        fft.process(&mut fft_input);

        // Detect vibrato characteristics
        let rate = self.detect_vibrato_rate(&fft_input)?;
        let depth = self.detect_vibrato_depth(&fft_input)?;
        let strength = self.calculate_vibrato_strength(rate, depth)?;

        Ok(VibratoFrame {
            rate,
            depth,
            strength,
            timestamp: frame_idx as f32 * self.hop_size as f32 / self.sample_rate as f32,
        })
    }

    /// Detect vibrato rate from frequency spectrum
    fn detect_vibrato_rate(&mut self, spectrum: &[Complex<f32>]) -> Result<f32> {
        let mut max_magnitude = 0.0;
        let mut peak_bin = 0;

        // Look for peaks in the vibrato frequency range
        let min_bin = ((self.config.frequency_range.0 * spectrum.len() as f32)
            / (self.sample_rate as f32 / 2.0)) as usize;
        let max_bin = ((self.config.frequency_range.1 * spectrum.len() as f32)
            / (self.sample_rate as f32 / 2.0)) as usize;

        for (bin, spectrum_value) in spectrum
            .iter()
            .enumerate()
            .take(max_bin.min(spectrum.len()))
            .skip(min_bin)
        {
            let magnitude = spectrum_value.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_bin = bin;
            }
        }

        // Convert bin to frequency
        let rate = (peak_bin as f32 * self.sample_rate as f32 / 2.0) / spectrum.len() as f32;
        Ok(rate)
    }

    /// Detect vibrato depth from frequency spectrum
    fn detect_vibrato_depth(&mut self, spectrum: &[Complex<f32>]) -> Result<f32> {
        // Calculate spectral centroid variation as depth indicator
        let mut centroid_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (bin, &complex) in spectrum.iter().enumerate() {
            let magnitude = complex.norm();
            let frequency = (bin as f32 * self.sample_rate as f32 / 2.0) / spectrum.len() as f32;

            centroid_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        let centroid = if magnitude_sum > 0.0 {
            centroid_sum / magnitude_sum
        } else {
            0.0
        };

        // Calculate variation from expected centroid
        let expected_centroid = 440.0; // A4 as reference
        let variation = (centroid - expected_centroid).abs();

        // Convert to cents (100 cents = 1 semitone)
        let depth = if expected_centroid > 0.0 {
            1200.0 * (variation / expected_centroid).ln() / 2.0_f32.ln()
        } else {
            0.0
        };

        Ok(depth
            .abs()
            .clamp(self.config.depth_range.0, self.config.depth_range.1))
    }

    /// Calculate vibrato strength
    fn calculate_vibrato_strength(&mut self, rate: f32, depth: f32) -> Result<f32> {
        // Normalize rate and depth to calculate strength
        let rate_norm = if self.config.frequency_range.1 > self.config.frequency_range.0 {
            (rate - self.config.frequency_range.0)
                / (self.config.frequency_range.1 - self.config.frequency_range.0)
        } else {
            0.0
        };

        let depth_norm = if self.config.depth_range.1 > self.config.depth_range.0 {
            (depth - self.config.depth_range.0)
                / (self.config.depth_range.1 - self.config.depth_range.0)
        } else {
            0.0
        };

        let strength = (rate_norm * 0.6 + depth_norm * 0.4).clamp(0.0, 1.0);
        Ok(strength)
    }

    /// Apply vibrato enhancement to frame
    fn apply_vibrato_enhancement(
        &mut self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        vibrato_info: &VibratoFrame,
    ) -> Result<()> {
        if vibrato_info.strength < self.config.detection_threshold {
            return Ok(());
        }

        let mel_bins = mel_spectrogram.shape()[0];
        let enhancement_factor = self.config.enhancement_strength * vibrato_info.strength;

        // Apply vibrato enhancement based on detected characteristics
        for bin_idx in 0..mel_bins {
            let frequency = self.mel_bin_to_hz(bin_idx, mel_bins);

            // Calculate vibrato modulation
            let modulation = self.calculate_vibrato_modulation(
                frequency,
                vibrato_info.rate,
                vibrato_info.depth,
                vibrato_info.timestamp,
            )?;

            // Apply enhancement
            let current_value = mel_spectrogram[[bin_idx, frame_idx]];
            let enhanced_value = current_value * (1.0 + modulation * enhancement_factor);

            mel_spectrogram[[bin_idx, frame_idx]] = enhanced_value;
        }

        Ok(())
    }

    /// Calculate vibrato modulation for a given frequency
    fn calculate_vibrato_modulation(
        &mut self,
        _frequency: f32,
        rate: f32,
        depth: f32,
        timestamp: f32,
    ) -> Result<f32> {
        // Calculate sinusoidal modulation
        let phase = 2.0 * std::f32::consts::PI * rate * timestamp;
        let modulation = (depth / 100.0) * phase.sin(); // Convert cents to ratio

        Ok(modulation)
    }

    /// Convert mel spectrogram frame to linear spectrum
    fn mel_to_spectrum(&mut self, frame: &ndarray::ArrayView1<f32>) -> Result<Vec<f32>> {
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
    fn apply_hann_window(&mut self, spectrum: &mut [f32]) {
        let n = spectrum.len();
        for (i, value) in spectrum.iter_mut().enumerate() {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            *value *= window;
        }
    }

    /// Convert mel frequency to Hz
    fn mel_to_hz(&mut self, mel: f32) -> f32 {
        700.0 * (mel / 1127.0).exp() - 700.0
    }

    /// Convert Hz to mel frequency
    fn hz_to_mel(&mut self, hz: f32) -> f32 {
        1127.0 * (1.0 + hz / 700.0).ln()
    }

    /// Convert mel bin index to Hz
    fn mel_bin_to_hz(&mut self, bin_idx: usize, mel_bins: usize) -> f32 {
        let mel_freq =
            (bin_idx as f32 / mel_bins as f32) * self.hz_to_mel(self.sample_rate as f32 / 2.0);
        self.mel_to_hz(mel_freq)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &VibratoConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }

    /// Get vibrato statistics
    pub fn get_vibrato_stats(&mut self) -> Result<VibratoStats> {
        if self.vibrato_history.is_empty() {
            return Ok(VibratoStats::default());
        }

        let rates: Vec<f32> = self.vibrato_history.iter().map(|v| v.rate).collect();
        let depths: Vec<f32> = self.vibrato_history.iter().map(|v| v.depth).collect();
        let strengths: Vec<f32> = self.vibrato_history.iter().map(|v| v.strength).collect();

        Ok(VibratoStats {
            average_rate: rates.iter().sum::<f32>() / rates.len() as f32,
            average_depth: depths.iter().sum::<f32>() / depths.len() as f32,
            average_strength: strengths.iter().sum::<f32>() / strengths.len() as f32,
            max_rate: rates.iter().fold(0.0, |a, &b| a.max(b)),
            max_depth: depths.iter().fold(0.0, |a, &b| a.max(b)),
            max_strength: strengths.iter().fold(0.0, |a, &b| a.max(b)),
        })
    }

    /// Reset vibrato history
    pub fn reset(&mut self) {
        self.vibrato_history.clear();
    }
}

/// Statistics for vibrato analysis
#[derive(Debug, Clone, Default)]
pub struct VibratoStats {
    /// Average vibrato rate
    pub average_rate: f32,
    /// Average vibrato depth
    pub average_depth: f32,
    /// Average vibrato strength
    pub average_strength: f32,
    /// Maximum vibrato rate
    pub max_rate: f32,
    /// Maximum vibrato depth
    pub max_depth: f32,
    /// Maximum vibrato strength
    pub max_strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_vibrato_processor_creation() {
        let config = VibratoConfig::default();
        let processor = VibratoProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_vibrato_analysis() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        // Create sample mel frame
        let frame = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.3, 0.1]);
        let frame_view = frame.view();

        let vibrato_info = processor.analyze_vibrato(&frame_view, 0);
        assert!(vibrato_info.is_ok());
    }

    #[test]
    fn test_vibrato_rate_detection() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        // Create sample spectrum
        let spectrum: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new((i as f32 / 1024.0).sin(), 0.0))
            .collect();

        let rate = processor.detect_vibrato_rate(&spectrum);
        assert!(rate.is_ok());
    }

    #[test]
    fn test_vibrato_depth_detection() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        // Create sample spectrum
        let spectrum: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new((i as f32 / 1024.0).sin(), 0.0))
            .collect();

        let depth = processor.detect_vibrato_depth(&spectrum);
        assert!(depth.is_ok());
    }

    #[test]
    fn test_vibrato_strength_calculation() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        let strength = processor.calculate_vibrato_strength(5.0, 50.0);
        assert!(strength.is_ok());
        let strength_val = strength.unwrap();
        assert!((0.0..=1.0).contains(&strength_val));
    }

    #[test]
    fn test_vibrato_modulation() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        let modulation = processor.calculate_vibrato_modulation(440.0, 5.0, 50.0, 0.1);
        assert!(modulation.is_ok());
    }

    #[test]
    fn test_process_mel_spectrogram() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = processor.process(&mel);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.shape(), mel.shape());
    }

    #[test]
    fn test_vibrato_stats() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        // Add sample vibrato frames
        processor.vibrato_history.push_back(VibratoFrame {
            rate: 5.0,
            depth: 50.0,
            strength: 0.8,
            timestamp: 0.0,
        });
        processor.vibrato_history.push_back(VibratoFrame {
            rate: 6.0,
            depth: 60.0,
            strength: 0.9,
            timestamp: 0.1,
        });

        let stats = processor.get_vibrato_stats();
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert_eq!(stats.average_rate, 5.5);
        assert_eq!(stats.average_depth, 55.0);
        assert_eq!(stats.average_strength, 0.85);
    }

    #[test]
    fn test_config_update() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        let new_config = VibratoConfig {
            enhancement_strength: 0.8,
            ..Default::default()
        };

        let result = processor.update_config(&new_config);
        assert!(result.is_ok());
        assert_eq!(processor.config.enhancement_strength, 0.8);
    }

    #[test]
    fn test_processor_reset() {
        let config = VibratoConfig::default();
        let mut processor = VibratoProcessor::new(&config).unwrap();

        processor.vibrato_history.push_back(VibratoFrame {
            rate: 5.0,
            depth: 50.0,
            strength: 0.8,
            timestamp: 0.0,
        });

        processor.reset();
        assert!(processor.vibrato_history.is_empty());
    }
}
