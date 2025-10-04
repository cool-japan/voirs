//! Breath sound processor for singing voice vocoder.

use crate::models::singing::config::BreathSoundConfig;
use anyhow::Result;
#[cfg(test)]
use scirs2_core::ndarray::Array1;
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::Complex;
use scirs2_fft::{FftPlanner, RealFftPlanner};

/// Processor for breath sound detection and enhancement/reduction
pub struct BreathSoundProcessor {
    /// Configuration
    config: BreathSoundConfig,
    /// Window size for analysis
    window_size: usize,
    /// Hop size for analysis
    hop_size: usize,
    /// Sample rate
    sample_rate: u32,
    /// Breath detection threshold
    breath_threshold: f32,
}

/// Breath sound analysis result
#[derive(Debug, Clone)]
pub struct BreathAnalysis {
    /// Breath sound strength (0.0-1.0)
    pub breath_strength: f32,
    /// Breath frequency content
    pub breath_spectrum: Vec<f32>,
    /// Breath/voice classification
    pub is_breath: bool,
    /// Breath sound type
    pub breath_type: BreathType,
}

/// Types of breath sounds
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BreathType {
    /// Inhalation breath
    Inhalation,
    /// Exhalation breath
    Exhalation,
    /// Fricative breath (consonants)
    Fricative,
    /// Aspirated breath (breathy voice)
    Aspirated,
    /// No breath detected
    None,
}

impl BreathSoundProcessor {
    /// Create new breath sound processor
    pub fn new(config: &BreathSoundConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            window_size: 2048,
            hop_size: 512,
            sample_rate: 22050,
            breath_threshold: config.detection_threshold,
        })
    }

    /// Process mel spectrogram for breath sound handling
    pub fn process(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.config.enable_processing {
            return Ok(mel_spectrogram.clone());
        }

        let mut processed = mel_spectrogram.clone();
        let frames = mel_spectrogram.shape()[1];

        // Process each frame
        for frame_idx in 0..frames {
            let frame = mel_spectrogram.column(frame_idx);
            let analysis = self.analyze_breath(&frame)?;

            // Apply breath sound processing
            self.apply_breath_processing(&mut processed, frame_idx, &analysis)?;
        }

        Ok(processed)
    }

    /// Continuous breath analysis with overlapping windows using hop_size
    /// This method uses the configured hop_size for sliding window analysis
    pub fn process_continuous(
        &mut self,
        mel_spectrogram: &Array2<f32>,
    ) -> Result<Vec<BreathAnalysis>> {
        if !self.config.enable_processing {
            return Ok(vec![]);
        }

        let frames = mel_spectrogram.shape()[1];
        let mut analyses = Vec::new();

        // Process with overlapping windows based on hop_size
        let hop_frames = self.hop_size / 256; // Convert audio hop_size to mel frame hops
        let window_frames = self.window_size / 256; // Convert audio window_size to mel frame window

        let mut frame_idx = 0;
        while frame_idx + window_frames <= frames {
            // Analyze window of frames
            let window_end = (frame_idx + window_frames).min(frames);
            let mut combined_analysis = BreathAnalysis {
                breath_strength: 0.0,
                breath_spectrum: vec![],
                is_breath: false,
                breath_type: BreathType::None,
            };

            // Average analysis over the window
            let mut total_strength = 0.0;
            let mut breath_count = 0;

            for window_frame in frame_idx..window_end {
                let frame = mel_spectrogram.column(window_frame);
                let analysis = self.analyze_breath(&frame)?;

                total_strength += analysis.breath_strength;
                if analysis.is_breath {
                    breath_count += 1;
                }

                // Use the strongest breath type detected in the window
                if analysis.breath_strength > combined_analysis.breath_strength {
                    combined_analysis.breath_type = analysis.breath_type;
                    combined_analysis.breath_spectrum = analysis.breath_spectrum;
                }
            }

            // Set combined analysis results
            combined_analysis.breath_strength = total_strength / window_frames as f32;
            combined_analysis.is_breath = breath_count > window_frames / 2;

            analyses.push(combined_analysis);

            // Move to next hop position
            frame_idx += hop_frames.max(1);
        }

        Ok(analyses)
    }

    /// Analyze breath sound characteristics in a frame
    fn analyze_breath(&mut self, frame: &ArrayView1<f32>) -> Result<BreathAnalysis> {
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

        // Analyze breath characteristics
        let breath_strength = self.calculate_breath_strength(&magnitude_spectrum)?;
        let breath_spectrum = self.extract_breath_spectrum(&magnitude_spectrum)?;
        let is_breath = breath_strength > self.breath_threshold;
        let breath_type = self.classify_breath_type(&magnitude_spectrum, breath_strength)?;

        Ok(BreathAnalysis {
            breath_strength,
            breath_spectrum,
            is_breath,
            breath_type,
        })
    }

    /// Calculate breath sound strength
    fn calculate_breath_strength(&mut self, spectrum: &[f32]) -> Result<f32> {
        let freq_range = &self.config.frequency_range;
        let min_bin =
            (freq_range.0 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let max_bin =
            (freq_range.1 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        // Calculate high-frequency energy (typical of breath sounds)
        let high_freq_energy: f32 = spectrum[min_bin..max_bin.min(spectrum.len())].iter().sum();

        // Calculate total energy
        let total_energy: f32 = spectrum.iter().sum();

        // Breath strength is ratio of high-frequency to total energy
        let breath_strength = if total_energy > 0.0 {
            high_freq_energy / total_energy
        } else {
            0.0
        };

        Ok(breath_strength.clamp(0.0, 1.0))
    }

    /// Extract breath-specific spectrum
    fn extract_breath_spectrum(&mut self, spectrum: &[f32]) -> Result<Vec<f32>> {
        let freq_range = &self.config.frequency_range;
        let min_bin =
            (freq_range.0 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let max_bin =
            (freq_range.1 * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        let breath_spectrum = spectrum[min_bin..max_bin.min(spectrum.len())].to_vec();
        Ok(breath_spectrum)
    }

    /// Classify breath sound type
    fn classify_breath_type(
        &mut self,
        spectrum: &[f32],
        breath_strength: f32,
    ) -> Result<BreathType> {
        if breath_strength < self.breath_threshold {
            return Ok(BreathType::None);
        }

        // Analyze spectral characteristics to classify breath type
        let spectral_centroid = self.calculate_spectral_centroid(spectrum)?;
        let spectral_rolloff = self.calculate_spectral_rolloff(spectrum)?;
        let spectral_flatness = self.calculate_spectral_flatness(spectrum)?;

        // Classification rules based on spectral features
        let breath_type = if spectral_flatness > 0.8 && spectral_centroid > 4000.0 {
            BreathType::Fricative
        } else if spectral_rolloff > 6000.0 && breath_strength > 0.7 {
            BreathType::Inhalation
        } else if spectral_rolloff > 5000.0 && breath_strength > 0.5 {
            BreathType::Exhalation
        } else if spectral_centroid > 2000.0 && breath_strength > 0.3 {
            BreathType::Aspirated
        } else {
            BreathType::None
        };

        Ok(breath_type)
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&mut self, spectrum: &[f32]) -> Result<f32> {
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

    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&mut self, spectrum: &[f32]) -> Result<f32> {
        let total_energy: f32 = spectrum.iter().sum();
        let threshold = total_energy * 0.85; // 85% rolloff point

        let mut cumulative_energy = 0.0;
        let mut rolloff_bin = 0;

        for (bin, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude;
            if cumulative_energy >= threshold {
                rolloff_bin = bin;
                break;
            }
        }

        let rolloff_freq =
            (rolloff_bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
        Ok(rolloff_freq)
    }

    /// Calculate spectral flatness
    fn calculate_spectral_flatness(&mut self, spectrum: &[f32]) -> Result<f32> {
        let positive_spectrum: Vec<f32> = spectrum.iter().filter(|&&x| x > 0.0).cloned().collect();

        if positive_spectrum.is_empty() {
            return Ok(0.0);
        }

        // Geometric mean
        let log_sum: f32 = positive_spectrum.iter().map(|x| x.ln()).sum();
        let geometric_mean = (log_sum / positive_spectrum.len() as f32).exp();

        // Arithmetic mean
        let arithmetic_mean =
            positive_spectrum.iter().sum::<f32>() / positive_spectrum.len() as f32;

        let flatness = if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        };

        Ok(flatness.clamp(0.0, 1.0))
    }

    /// Apply breath sound processing to frame
    fn apply_breath_processing(
        &mut self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        analysis: &BreathAnalysis,
    ) -> Result<()> {
        if !analysis.is_breath {
            return Ok(());
        }

        let mel_bins = mel_spectrogram.shape()[0];
        let freq_range = self.config.frequency_range;

        // Calculate mel bin range for breath frequencies
        let min_mel_bin = self.hz_to_mel_bin(freq_range.0, mel_bins);
        let max_mel_bin = self.hz_to_mel_bin(freq_range.1, mel_bins);

        // Apply processing based on breath type
        match analysis.breath_type {
            BreathType::Inhalation | BreathType::Exhalation => {
                // Reduce natural breath sounds
                self.apply_breath_reduction(mel_spectrogram, frame_idx, min_mel_bin, max_mel_bin)?;
            }
            BreathType::Fricative => {
                // Enhance fricative consonants
                self.apply_breath_enhancement(
                    mel_spectrogram,
                    frame_idx,
                    min_mel_bin,
                    max_mel_bin,
                )?;
            }
            BreathType::Aspirated => {
                // Moderate processing for breathy voice
                self.apply_moderate_processing(
                    mel_spectrogram,
                    frame_idx,
                    min_mel_bin,
                    max_mel_bin,
                )?;
            }
            BreathType::None => {
                // No processing needed
            }
        }

        Ok(())
    }

    /// Apply breath reduction
    fn apply_breath_reduction(
        &mut self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        min_bin: usize,
        max_bin: usize,
    ) -> Result<()> {
        let reduction_factor = 1.0 - self.config.reduction_strength;

        for bin_idx in min_bin..max_bin {
            if bin_idx < mel_spectrogram.shape()[0] {
                let current_value = mel_spectrogram[[bin_idx, frame_idx]];
                mel_spectrogram[[bin_idx, frame_idx]] = current_value * reduction_factor;
            }
        }

        Ok(())
    }

    /// Apply breath enhancement
    fn apply_breath_enhancement(
        &mut self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        min_bin: usize,
        max_bin: usize,
    ) -> Result<()> {
        let enhancement_factor = 1.0 + self.config.enhancement_strength;

        for bin_idx in min_bin..max_bin {
            if bin_idx < mel_spectrogram.shape()[0] {
                let current_value = mel_spectrogram[[bin_idx, frame_idx]];
                mel_spectrogram[[bin_idx, frame_idx]] = current_value * enhancement_factor;
            }
        }

        Ok(())
    }

    /// Apply moderate processing for breathy voice
    fn apply_moderate_processing(
        &mut self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        min_bin: usize,
        max_bin: usize,
    ) -> Result<()> {
        let moderate_factor = 1.0 + (self.config.enhancement_strength * 0.3);

        for bin_idx in min_bin..max_bin {
            if bin_idx < mel_spectrogram.shape()[0] {
                let current_value = mel_spectrogram[[bin_idx, frame_idx]];
                mel_spectrogram[[bin_idx, frame_idx]] = current_value * moderate_factor;
            }
        }

        Ok(())
    }

    /// Convert mel frame to linear spectrum
    fn mel_to_linear_spectrum(&mut self, frame: &ArrayView1<f32>) -> Result<Vec<f32>> {
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
    fn apply_hann_window(&mut self, spectrum: &[f32]) -> Vec<f32> {
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
    fn mel_to_hz(&mut self, mel: f32) -> f32 {
        700.0 * (mel / 1127.0).exp() - 700.0
    }

    /// Convert Hz to mel frequency
    fn hz_to_mel(&mut self, hz: f32) -> f32 {
        1127.0 * (1.0 + hz / 700.0).ln()
    }

    /// Convert Hz to mel bin index
    fn hz_to_mel_bin(&mut self, hz: f32, mel_bins: usize) -> usize {
        let mel_freq = self.hz_to_mel(hz);
        let max_mel = self.hz_to_mel(self.sample_rate as f32 / 2.0);
        let bin_idx = (mel_freq / max_mel * mel_bins as f32) as usize;
        bin_idx.min(mel_bins - 1)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &BreathSoundConfig) -> Result<()> {
        self.config = config.clone();
        self.breath_threshold = config.detection_threshold;
        Ok(())
    }

    /// Get breath sound statistics
    pub fn get_breath_stats(&mut self, analysis: &BreathAnalysis) -> BreathStats {
        BreathStats {
            breath_strength: analysis.breath_strength,
            breath_type: analysis.breath_type,
            is_breath: analysis.is_breath,
            breath_frequency_energy: analysis.breath_spectrum.iter().sum::<f32>(),
            breath_frequency_peak: analysis
                .breath_spectrum
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as f32)
                .unwrap_or(0.0),
        }
    }
}

/// Statistics for breath sound analysis
#[derive(Debug, Clone)]
pub struct BreathStats {
    /// Breath sound strength
    pub breath_strength: f32,
    /// Breath sound type
    pub breath_type: BreathType,
    /// Whether breath is detected
    pub is_breath: bool,
    /// Total energy in breath frequency range
    pub breath_frequency_energy: f32,
    /// Peak frequency in breath range
    pub breath_frequency_peak: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breath_sound_processor_creation() {
        let config = BreathSoundConfig::default();
        let processor = BreathSoundProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_breath_analysis() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        // Create sample mel frame
        let frame = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.8, 0.9]); // High-frequency emphasis
        let frame_view = frame.view();

        let analysis = processor.analyze_breath(&frame_view);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_breath_strength_calculation() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        // Create spectrum with high-frequency content
        let spectrum = vec![0.1, 0.1, 0.1, 0.8, 0.9, 0.8, 0.7];

        let strength = processor.calculate_breath_strength(&spectrum);
        assert!(strength.is_ok());
        assert!(strength.unwrap() > 0.0);
    }

    #[test]
    fn test_breath_type_classification() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        // Create spectrum with different characteristics
        let spectrum = vec![0.1; 1024];

        let breath_type = processor.classify_breath_type(&spectrum, 0.6);
        assert!(breath_type.is_ok());
    }

    #[test]
    fn test_spectral_features() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        let spectrum = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1];

        let centroid = processor.calculate_spectral_centroid(&spectrum);
        assert!(centroid.is_ok());

        let rolloff = processor.calculate_spectral_rolloff(&spectrum);
        assert!(rolloff.is_ok());

        let flatness = processor.calculate_spectral_flatness(&spectrum);
        assert!(flatness.is_ok());
    }

    #[test]
    fn test_breath_processing() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = processor.process(&mel);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.shape(), mel.shape());
    }

    #[test]
    fn test_breath_reduction() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        let mut mel = Array2::ones((80, 100));
        let original_value = mel[[40, 50]];

        let result = processor.apply_breath_reduction(&mut mel, 50, 30, 60);
        assert!(result.is_ok());

        // Value should be reduced
        assert!(mel[[40, 50]] < original_value);
    }

    #[test]
    fn test_breath_enhancement() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        let mut mel = Array2::ones((80, 100));
        let original_value = mel[[40, 50]];

        let result = processor.apply_breath_enhancement(&mut mel, 50, 30, 60);
        assert!(result.is_ok());

        // Value should be enhanced
        assert!(mel[[40, 50]] > original_value);
    }

    #[test]
    fn test_config_update() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        let new_config = BreathSoundConfig {
            detection_threshold: 0.1,
            ..Default::default()
        };

        let result = processor.update_config(&new_config);
        assert!(result.is_ok());
        assert_eq!(processor.config.detection_threshold, 0.1);
        assert_eq!(processor.breath_threshold, 0.1);
    }

    #[test]
    fn test_breath_stats() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        let analysis = BreathAnalysis {
            breath_strength: 0.7,
            breath_spectrum: vec![0.1, 0.5, 0.8, 0.3],
            is_breath: true,
            breath_type: BreathType::Fricative,
        };

        let stats = processor.get_breath_stats(&analysis);
        assert_eq!(stats.breath_strength, 0.7);
        assert_eq!(stats.breath_type, BreathType::Fricative);
        assert!(stats.is_breath);
        assert!(stats.breath_frequency_energy > 0.0);
    }

    #[test]
    fn test_breath_spectrum_extraction() {
        let config = BreathSoundConfig::default();
        let mut processor = BreathSoundProcessor::new(&config).unwrap();

        let spectrum = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1];

        let breath_spectrum = processor.extract_breath_spectrum(&spectrum);
        assert!(breath_spectrum.is_ok());
        assert!(!breath_spectrum.unwrap().is_empty());
    }
}
